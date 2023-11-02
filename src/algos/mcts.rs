use crate::core::position;
use smallvec::SmallVec;

struct Node<Action: Clone, Position: position::Position<Action>> {
    /// Action that led to this state
    action: Option<Action>,
    /// Game state
    position: Position,
    /// Child nodes
    children: SmallVec<[usize; 1]>,
    /// Fully expanded
    fully_expanded: bool,

    /// Accumulated value
    w: f64,
    /// Number of visits
    n: u32,
}

pub struct MCTS<Action: Clone, Position: position::Position<Action>> {
    nodes: Vec<Node<Action, Position>>,
}

impl<Action: Clone, Position: position::Position<Action>> MCTS<Action, Position> {
    pub fn new(position: &Position) -> Self {
        let root = Node {
            action: None,
            position: position.clone(),
            children: SmallVec::new(),
            fully_expanded: position.valid_actions().is_empty(),
            w: 0.0,
            n: 0,
        };

        MCTS { nodes: vec![root] }
    }

    pub fn run_iteration(&mut self) {
        let selected_index = self.select(0); // start at the root
        let expanded_index = self.expand(selected_index);
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

    fn expand(&mut self, index: usize) {
        let node = &mut self.nodes[index];

        // get next action; respect the order defined by `valid_actions`
        let action = node
            .position
            .valid_actions()
            .get(node.children.len())
            .unwrap()
            .clone();

        // generate new node
        let new_position = node.position.apply_action(&action);
        let new_is_fully_expanded = new_position.valid_actions().is_empty();
        let new_node = Node {
            action: Some(action.clone()),
            position: new_position,
            children: SmallVec::from_elem(self.nodes.len(), 1),
            fully_expanded: new_is_fully_expanded,
            w: 0.0,
            n: 0,
        };

        // add the new node to the tree
        self.nodes.push(new_node);
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
