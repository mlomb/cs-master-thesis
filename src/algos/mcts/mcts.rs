use crate::core::{
    outcome::Outcome,
    position::{self, Position},
};
use smallvec::SmallVec;

use super::strategy::{self, Strategy};

pub trait MCTSSpec: Sized {
    type Position: Position;
    type Strategy: strategy::Strategy<Self::Position> + Clone;
}

struct Node<Position: position::Position> {
    /// Action that led to this state
    action: Option<Position::Action>,
    /// Game state
    position: Position,
    /// Parent node
    parent: Option<usize>,
    /// Child nodes
    children: SmallVec<[usize; 1]>,
    /// Position status
    status: Option<Outcome>,
    /// Fully expanded
    fully_expanded: bool,

    /// Accumulated value
    w: f64,
    /// Number of visits
    n: u32,
}

/// Monte Carlo Tree Search
pub struct MCTS<Spec: MCTSSpec> {
    /// Root node index
    ///
    /// It is useful to change the root to reuse previous iterations.
    /// This could be changed to also prune the tree and keep the root at 0,
    /// but that would require readjusting the indices of all nodes.
    root_index: usize,

    /// The list of nodes in the tree
    nodes: Vec<Node<Spec::Position>>,

    /// TODO: change name
    strategy: Spec::Strategy,
}

impl<Spec: MCTSSpec> MCTS<Spec> {
    pub fn new(position: &Spec::Position, strategy: &Spec::Strategy) -> Self {
        let root = Node {
            action: None,
            position: position.clone(),
            parent: None,
            children: SmallVec::new(),
            status: position.status(),
            fully_expanded: position.valid_actions().is_empty(),
            w: 0.0,
            n: 0,
        };

        MCTS {
            root_index: 0,
            nodes: vec![root],
            strategy: strategy.clone(),
        }
    }

    pub fn run_iteration(&mut self) {
        let selected_index = self.select(self.root_index);
        let index: usize;

        if self.nodes[selected_index].status.is_some() {
            // the node is terminal
            index = selected_index;
        } else {
            // the node is not terminal, so expand it first
            index = self.expand(selected_index);
        }

        self.backprop(index, self.strategy.rollout(&self.nodes[index].position));
    }

    fn select(&self, index: usize) -> usize {
        let node = &self.nodes[index];

        if node.status.is_some() {
            // the node is terminal, so return it
            return index;
        }

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
        let new_status = new_position.status();
        let new_fully_expanded = new_position.valid_actions().is_empty();
        let new_node = Node {
            action: Some(action.clone()),
            position: new_position,
            parent: Some(index),
            children: SmallVec::new(),
            status: new_status,
            fully_expanded: new_fully_expanded,
            w: 0.0,
            n: 0,
        };

        node.children.push(new_index);
        node.fully_expanded = node.children.len() == actions.len();
        self.nodes.push(new_node);

        new_index
    }

    fn backprop(&mut self, from_index: usize, new_value: f64) {
        let mut current_index = Some(from_index);
        let mut value = new_value;

        while let Some(index) = current_index {
            self.nodes[index].n += 1;
            self.nodes[index].w += value;

            current_index = self.nodes[index].parent;
            value = -value;
        }
    }

    /// Changes the root node to the child corresponding to the given action.
    pub fn move_root(
        &mut self,
        action: &<<Spec as MCTSSpec>::Position as position::Position>::Action,
    ) {
        let root = &self.nodes[self.root_index];
        let child_index = root
            .children
            .iter()
            .find(|&&idx| self.nodes[idx].action.as_ref() == Some(&action));

        match child_index {
            None => todo!(),
            Some(index) => self.root_index = *index,
        }
    }

    pub fn get_action_distribution(
        &self,
    ) -> Vec<(
        // TODO: change for something nicer
        <<Spec as MCTSSpec>::Position as position::Position>::Action,
        f64,
    )> {
        let root = &self.nodes[self.root_index];
        root.children
            .iter()
            .map(|&idx| {
                let node = &self.nodes[idx];
                (
                    node.action.clone().unwrap(),
                    (node.n as f64) / (root.n as f64),
                )
            })
            .collect()
    }

    pub fn get_best_action(
        &self,
    ) -> Option<<<Spec as MCTSSpec>::Position as position::Position>::Action> {
        self.get_action_distribution()
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(action, _)| action)
    }
}

impl<Position: position::Position> Node<Position> {
    fn ucb1(&self, parent_n: u32) -> f64 {
        let c = 1.0;
        let n = self.n as f64;
        let w = self.w;
        let t = parent_n as f64;
        w / n + c * (t.ln() / n).sqrt()
    }
}
