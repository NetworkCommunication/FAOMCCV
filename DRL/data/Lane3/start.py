import csv
import os
import sys
import traci
import traci.constants as tc

sumo_binary = "sumo-gui"
sumocfg_file = "StraightRoad.sumocfg"
net_file = "StraightRoad.net.xml"
route_file = "mixed_StraightRoad.rou.xml"

sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "1", "--scale","1"]
traci.start(sumo_cmd)
output_file = "vehicle_data2.csv"

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Vehicle_ID', 'Distance_from_Start', 'Lane_ID', 'Vehicle_Speed', 'Vehicle_acceleration'])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        arrived_vehicles = traci.simulation.getArrivedIDList()

        if arrived_vehicles:
            break

        vehicles = traci.vehicle.getIDList()

        for vehicle_id in vehicles:
            time = traci.simulation.getTime()
            distance_from_start = traci.vehicle.getLanePosition(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id).split('_')[-1]
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            vehicle_acceleration = traci.vehicle.getAcceleration(vehicle_id)

            writer.writerow([time, vehicle_id, distance_from_start, lane_id, vehicle_speed, vehicle_acceleration])

    traci.close()
