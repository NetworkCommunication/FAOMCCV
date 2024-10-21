import numpy as np

time_step = 0.5  # Example time step
# Define the vehicle class
class Vehicle:
    def __init__(self, id, lane, position, velocity, acceleration):
        self.id = id
        self.lane = lane  # Lateral lane number
        self.position = position  # Longitudinal position
        self.velocity = velocity
        self.acceleration = acceleration

# Define the maneuver class
class Maneuver:
    def __init__(self, lateral_behavior, longitudinal_behavior):
        self.lateral_behavior = lateral_behavior  # Blc: {Lt, Rt, Nt}
        self.longitudinal_behavior = longitudinal_behavior  # Bm: {Up, Dw, Mt}

# Define the node class for the maneuver tree
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.benefit = 0  # VA
        self.autonomous_vehicle = None
        self.conventional_vehicles = None

# Define the tree class for the maneuver tree
class ManeuverTree:
    def __init__(self, depth, autonomous_vehicle, conventional_vehicles):
        self.root = Node(state=None)
        self.depth = depth
        self.root.autonomous_vehicle = autonomous_vehicle
        self.root.conventional_vehicles = conventional_vehicles

    def expand_node(self, node, current_lane, is_root=False):
        if is_root:
            if current_lane == 2:
                maneuvers = [("Nt", "Up"), ("Nt", "Dw"), ("Nt", "Mt")]
            else:
                maneuvers = [("Lt", "Up"), ("Lt", "Dw"), ("Lt", "Mt"),
                             ("Nt", "Up"), ("Nt", "Dw"), ("Nt", "Mt")]
        else:
            if current_lane == 0:
                maneuvers = [("Lt", "Up"), ("Lt", "Dw"), ("Lt", "Mt"),
                             ("Nt", "Up"), ("Nt", "Dw"), ("Nt", "Mt")]
            elif current_lane == 2:
                maneuvers = [("Nt", "Up"), ("Nt", "Dw"), ("Nt", "Mt"),
                             ("Rt", "Up"), ("Rt", "Dw"), ("Rt", "Mt")]
            else:
                maneuvers = [("Lt", "Up"), ("Lt", "Dw"), ("Lt", "Mt"),
                             ("Nt", "Up"), ("Nt", "Dw"), ("Nt", "Mt"),
                             ("Rt", "Up"), ("Rt", "Dw"), ("Rt", "Mt")]

        for maneuver in maneuvers:
            maneuver_obj = Maneuver(*maneuver)
            child_node = Node(state=maneuver_obj, parent=node)
            child_node.autonomous_vehicle = update_autonomous_vehicle_state(node.autonomous_vehicle, maneuver_obj)
            child_node.conventional_vehicles = update_conventional_vehicles_state(node.conventional_vehicles)
            node.children.append(child_node)

    def build_tree(self):
        nodes = [self.root]
        is_root = True
        for _ in range(self.depth):
            new_nodes = []
            for node in nodes:
                current_lane = node.autonomous_vehicle.lane
                self.expand_node(node, current_lane, is_root)
                new_nodes.extend(node.children)
            nodes = new_nodes
            is_root = False
        return self.root

# Define the adaptive beam search algorithm
def adaptive_beam_search(tree, threshold):
    def compute_cumulative_weight(node):
        if node.parent is None:
            return node.benefit
        return node.benefit + compute_cumulative_weight(node.parent)

    root = tree.root
    current_nodes = [root]
    for _ in range(tree.depth):
        next_nodes = []
        for node in current_nodes:
            for child in node.children:
                child.benefit, child.beneVel, child.beneETTC = compute_benefit(node.autonomous_vehicle, node.conventional_vehicles, child.state)
                next_nodes.append(child)
        next_nodes.sort(key=compute_cumulative_weight, reverse=True)
        high_confidence_nodes = [node for node in next_nodes if compute_cumulative_weight(node) - compute_cumulative_weight(next_nodes[0]) < threshold]

        for node in high_confidence_nodes:
            if node.parent is not None:
                node.autonomous_vehicle = update_autonomous_vehicle_state(node.parent.autonomous_vehicle, node.state)
                node.conventional_vehicles = update_conventional_vehicles_state(node.parent.conventional_vehicles)

        current_nodes = high_confidence_nodes
    return max(current_nodes, key=compute_cumulative_weight)

def update_autonomous_vehicle_state(pre_autonomous_vehicle, maneuver):
    v_max = 30
    updated_vehicle = Vehicle(
        id=pre_autonomous_vehicle.id,
        lane=pre_autonomous_vehicle.lane,
        position=pre_autonomous_vehicle.position,
        velocity=pre_autonomous_vehicle.velocity,
        acceleration=pre_autonomous_vehicle.acceleration
    )
    if maneuver.lateral_behavior == "Lt":
        updated_vehicle.lane = min(updated_vehicle.lane + 1, 2)
    elif maneuver.lateral_behavior == "Rt":
        updated_vehicle.lane = max(updated_vehicle.lane - 1, 0)

    if maneuver.longitudinal_behavior == "Up":
        updated_vehicle.velocity = min(
            updated_vehicle.velocity + abs(updated_vehicle.acceleration) * time_step,
            v_max)
        updated_vehicle.position = updated_vehicle.position + updated_vehicle.velocity * time_step + 1 / 2 * abs(
            updated_vehicle.acceleration) * time_step ** 2
    elif maneuver.longitudinal_behavior == "Dw":
        updated_vehicle.velocity = max(
            updated_vehicle.velocity - abs(updated_vehicle.acceleration) * time_step, 0)
        updated_vehicle.position = updated_vehicle.position + updated_vehicle.velocity * time_step - 1 / 2 * abs(
            updated_vehicle.acceleration) * time_step ** 2

    return updated_vehicle

def update_conventional_vehicles_state(pre_conventional_vehicles):
    updated_vehicles = []
    for vehicle in pre_conventional_vehicles:
        updated_vehicle = Vehicle(
            id=vehicle.id,
            lane=vehicle.lane,
            position=vehicle.position + vehicle.velocity * time_step,
            velocity=vehicle.velocity,
            acceleration=vehicle.acceleration
        )
        updated_vehicles.append(updated_vehicle)
    return updated_vehicles

def compute_ETTC(autonomous_vehicle, conventional_vehicles):
    vehicleLength = 5
    deltaD = conventional_vehicles.position - autonomous_vehicle.position - vehicleLength
    deltaV = conventional_vehicles.velocity - autonomous_vehicle.velocity
    if deltaD >= 0:
        if deltaV < 0:
            ETTC = abs(deltaD / deltaV)
        else:
            ETTC = 10
    else:
        if deltaV > 0:
            ETTC = abs(deltaD / deltaV)
        else:
            ETTC = 10
    return min(ETTC, 10)

# Placeholder function to compute the benefit VA
def compute_benefit(autonomous_vehicle, conventional_vehicles, maneuver):
    # Implement the logic to compute the benefit of the maneuver
    v_max = 30
    ETTC_max = 10

    new_autonomous_vehicle = Vehicle(
        id=autonomous_vehicle.id,
        lane=autonomous_vehicle.lane,
        position=autonomous_vehicle.position,
        velocity=autonomous_vehicle.velocity,
        acceleration=autonomous_vehicle.acceleration
    )

    if maneuver.lateral_behavior == "Lt":
        new_autonomous_vehicle.lane = min(new_autonomous_vehicle.lane + 1, 2)
    elif maneuver.lateral_behavior == "Rt":
        new_autonomous_vehicle.lane = max(new_autonomous_vehicle.lane - 1, 0)

    if maneuver.longitudinal_behavior == "Up":
        new_autonomous_vehicle.velocity = min(new_autonomous_vehicle.velocity + abs(new_autonomous_vehicle.acceleration) * time_step, v_max)  # 加速，不能超过最大速度
        new_autonomous_vehicle.position = new_autonomous_vehicle.position + new_autonomous_vehicle.velocity * time_step + 1/2 * abs(new_autonomous_vehicle.acceleration) * time_step**2
    elif maneuver.longitudinal_behavior == "Dw":
        new_autonomous_vehicle.velocity = max(new_autonomous_vehicle.velocity - abs(new_autonomous_vehicle.acceleration) * time_step, 0)  # 减速，不能低于0
        new_autonomous_vehicle.position = new_autonomous_vehicle.position + new_autonomous_vehicle.velocity * time_step - 1 / 2 * abs(
            new_autonomous_vehicle.acceleration) * time_step ** 2

    new_conventional_vehicles = []
    for vehicle in conventional_vehicles:
        new_vehicle = Vehicle(
            id=vehicle.id,
            lane=vehicle.lane,
            position=vehicle.position + vehicle.velocity * time_step,
            velocity=vehicle.velocity,
            acceleration=vehicle.acceleration
        )
        new_conventional_vehicles.append(new_vehicle)

    beneVel = autonomous_vehicle.velocity / v_max

    associated_vehicles = []
    if maneuver.lateral_behavior == "Lt":
        associated_vehicles = [v for v in new_conventional_vehicles if v.lane == new_autonomous_vehicle.lane]
    elif maneuver.lateral_behavior == "Rt":
        associated_vehicles = [v for v in new_conventional_vehicles if v.lane == new_autonomous_vehicle.lane]
    else:
        associated_vehicles = [v for v in new_conventional_vehicles if
                               v.lane == new_autonomous_vehicle.lane and v.position > new_autonomous_vehicle.position]

    min_ETTC = 10
    for vehicle in associated_vehicles:
        ETTC = compute_ETTC(autonomous_vehicle, vehicle)
        if ETTC < min_ETTC:
            min_ETTC = ETTC

    beneETTC = min_ETTC / ETTC_max

    benefit = beneVel + beneETTC
    # print(f"Benefit: {benefit}, beneVel: {beneVel}, beneETTC: {beneETTC}")

    return benefit, beneVel, beneETTC  # Placeholder return value


if __name__ == '__main__':

    # Parameters


    # Example data for autonomous vehicle and conventional vehicles
    autonomous_vehicle = Vehicle(id='AV', lane=1, position=100, velocity=20, acceleration=2)  # Example position

    conventional_vehicles = [
        Vehicle(id='CV1', lane=1, position=105, velocity=20, acceleration=0),
        Vehicle(id='CV2', lane=1, position=95, velocity=20, acceleration=0),
        Vehicle(id='CV3', lane=2, position=95, velocity=20, acceleration=0),
        Vehicle(id='CV4', lane=0, position=105, velocity=20, acceleration=0),
        Vehicle(id='CV4', lane=2, position=115, velocity=20, acceleration=0),
    ]

    # Build the maneuver tree and perform adaptive beam search
    tree = ManeuverTree(depth=5, autonomous_vehicle=autonomous_vehicle, conventional_vehicles=conventional_vehicles)
    tree.build_tree()
    optimal_node = adaptive_beam_search(tree, threshold=0.4)

    def trace_optimal_path(node):
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    optimal_path = trace_optimal_path(optimal_node)

    print("Optimal path from root to leaf:")
    for node in optimal_path:
        if node.state is not None:
            print(
                f"Maneuver: {node.state.lateral_behavior}, {node.state.longitudinal_behavior}, Benefit: {node.benefit}, beneVel: {node.beneVel}, beneETTC: {node.beneETTC}")
