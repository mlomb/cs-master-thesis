use std::marker::PhantomData;

use crate::core::{
    position,
    value::{self, Value},
};
use smallvec::SmallVec;

use super::strategy;

struct Node<Action: Clone, Position: position::Position<Action>> {
    /// Action that led to this state
    action: Option<Action>,
    /// Game state
    position: Position,
    /// Parent node
    parent: Option<usize>,
    /// Child nodes
    children: SmallVec<[usize; 1]>,
    /// Fully expanded
    fully_expanded: bool,

    /// Accumulated value
    w: f64,
    /// Number of visits
    n: u32,
}

pub struct MCTS<Action, Position, Strategy>
where
    Action: Clone,
    Position: position::Position<Action>,
    Strategy: strategy::Strategy<Action, Position>,
{
    nodes: Vec<Node<Action, Position>>,

    strategy: Strategy,
}

impl<Action, Position, Strategy> MCTS<Action, Position, Strategy>
where
    Action: Clone,
    Position: position::Position<Action>,
    Strategy: strategy::Strategy<Action, Position> + Clone,
{
    pub fn new(position: &Position, strategy: &Strategy) -> Self {
        let root = Node {
            action: None,
            position: position.clone(),
            parent: None,
            children: SmallVec::new(),
            fully_expanded: position.valid_actions().is_empty(),
            w: 0.0,
            n: 0,
        };

        MCTS {
            nodes: vec![root],
            strategy: strategy.clone(),
        }
    }

    pub fn run_iteration(&mut self) {
        let selected_index = self.select(0); // start at the root
        let expanded_index = self.expand(selected_index);
        let rollout_value = self.strategy.rollout(&self.nodes[expanded_index].position);
        self.backprop(expanded_index, rollout_value);
    }

    fn select(&self, index: usize) -> usize {
        let node = &self.nodes[index];

        if node.fully_expanded {
            // select the child with the highest UCB1 value
            let (child_index, _) = node
                .children
                .iter()
                .map(|&idx| (idx, self.nodes[idx].ucb1(node.n)))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .expect("At least one children");

            // continue selection
            self.select(child_index)
        } else {
            // the node is missing an edge, so select this to expand
            index
        }
    }

    fn expand(&mut self, index: usize) -> usize {
        let new_index = self.nodes.len();
        let node = &mut self.nodes[index];

        // get next action; respect the order defined by `valid_actions`
        let actions = node.position.valid_actions();
        let action = actions.get(node.children.len()).unwrap().clone();

        let new_position = node.position.apply_action(&action);
        let new_is_fully_expanded = new_position.valid_actions().is_empty();
        let new_node = Node {
            action: Some(action.clone()),
            position: new_position,
            parent: Some(index),
            children: SmallVec::new(),
            fully_expanded: new_is_fully_expanded,
            w: 0.0,
            n: 0,
        };

        node.children.push(new_index);
        node.fully_expanded = node.children.len() == actions.len();
        self.nodes.push(new_node);

        new_index
    }

    fn backprop(&mut self, from_index: usize, new_value: f64) {
        let mut index = from_index;
        let mut value = new_value;

        while let Some(parent_index) = self.nodes[index].parent {
            self.nodes[index].n += 1;
            self.nodes[index].w = self.strategy.backprop(self.nodes[index].w, value);
            index = parent_index;
            value *= -1.0;
        }
    }
}

impl<Action: Clone, Position: position::Position<Action>> Node<Action, Position> {
    fn ucb1(&self, parent_n: u32) -> f64 {
        let c = 1.0;
        let n = self.n as f64;
        let w = self.w;
        let t = parent_n as f64;
        w / n + c * (t.ln() / n).sqrt()
    }
}
