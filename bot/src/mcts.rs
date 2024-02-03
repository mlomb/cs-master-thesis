use crate::position::{GamePosition, Outcome};
use smallvec::SmallVec;

type Position = shakmaty::Chess;
type Action = <Position as GamePosition>::Action;
type Value = f32;

pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> Value;
}

struct Node {
    /// Action that led to this state
    action: Option<Action>,
    /// Game state
    position: Position,
    /// Parent node
    parent: Option<usize>,
    /// Child nodes
    children: SmallVec<[usize; 8]>,
    /// Position status
    status: Option<Outcome>,
    /// Fully expanded
    fully_expanded: bool,

    /// Accumulated value
    w: Value,
    /// Number of visits
    n: u32,
}

/// Monte Carlo Tree Search
pub struct MCTS<'a> {
    /// Root node index
    ///
    /// It is useful to change the root to reuse previous iterations.
    /// This could be changed to also prune the tree and keep the root at 0,
    /// but that would require readjusting the indices of all nodes.
    root_index: usize,

    /// The list of nodes in the tree
    nodes: Vec<Node>,

    /// Buffer for actions
    actions_buffer: Vec<Action>,

    evaluator: &'a dyn Evaluator,
}

impl<'a> MCTS<'a> {
    pub fn new(position: &Position, evaluator: &'a dyn Evaluator) -> Self {
        let root = Node {
            action: None,
            position: position.clone(),
            parent: None,
            children: SmallVec::new(),
            status: position.status(),
            fully_expanded: false, //  position.valid_actions().is_empty()
            w: 0.0,
            n: 0,
        };

        let mut nodes = vec![root];
        nodes.reserve(10_000_000);

        MCTS {
            root_index: 0,
            nodes,
            actions_buffer: Vec::with_capacity(128),
            evaluator,
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

        self.backprop(index, self.evaluator.evaluate(&self.nodes[index].position));
    }

    #[inline]
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

    #[inline]
    fn expand(&mut self, index: usize) -> usize {
        let new_index = self.nodes.len();
        let node = &mut self.nodes[index];

        // get next action; respect the order defined by `valid_actions`
        self.actions_buffer.clear();
        node.position.valid_actions(&mut self.actions_buffer);
        let num_parent_actions = self.actions_buffer.len();
        let action = self
            .actions_buffer
            .get(node.children.len())
            .unwrap()
            .clone();

        let new_position = node.position.apply_action(&action);
        let new_status = new_position.status();
        self.actions_buffer.clear();
        new_position.valid_actions(&mut self.actions_buffer);
        let new_node = Node {
            action: Some(action.clone()),
            position: new_position,
            parent: Some(index),
            children: SmallVec::new(),
            status: new_status,
            fully_expanded: self.actions_buffer.is_empty(),
            w: 0.0,
            n: 0,
        };

        node.children.push(new_index);
        node.fully_expanded = node.children.len() == num_parent_actions;
        self.nodes.push(new_node);

        new_index
    }

    #[inline]
    fn backprop(&mut self, from_index: usize, new_value: Value) {
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
    pub fn move_root(&mut self, action: &Action) {
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

    pub fn get_action_distribution(&self) -> Vec<(Action, f64)> {
        let root = &self.nodes[self.root_index];
        root.children
            .iter()
            .map(|&idx| {
                let node = &self.nodes[idx];
                (
                    node.action.clone().unwrap(),
                    (node.n as f64) / (root.n as f64),
                    //(node.w as f64) / (node.n as f64),
                )
            })
            .collect()
    }

    pub fn get_best_action(&self) -> Option<Action> {
        self.get_action_distribution()
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(action, _)| action)
    }
}

impl Node {
    #[inline]
    fn ucb1(&self, parent_n: u32) -> Value {
        let c = 1.4;
        let n = self.n as f32;
        let w = self.w;
        let t = parent_n as f32;
        w / n + c * (t.ln() / n).sqrt()
    }
}
