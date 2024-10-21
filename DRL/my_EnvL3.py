from __future__ import absolute_import
from __future__ import print_function
import logging
import math
import time

import gym
import numpy as np
import pandas as pd
from gym import spaces
import random as rn
import os
import sys
import traci
import traci.constants as tc
import torch.nn.functional as F

# we need to import python modules from the $SUMO_HOME/tools directory
from DRL.bayesNetworkGMM import gaussianProcessing, load_model, prediction
from section2.searchLaneChange.searchManeuerSequences import Vehicle, ManeuverTree, adaptive_beam_search

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

gui = False
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')

config_path = "data/Lane3/StraightRoad.sumocfg"

class LaneChangePredict(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.minAutoVelocity = 0
        self.maxAutoVelocity = 30

        self.minDistanceFrontVeh = 0
        self.maxDistanceFrontVeh = 150

        self.minDistanceRearVeh = 0
        self.maxDistanceRearVeh = 150

        self.minLaneNumber = 0
        self.maxLaneNumber = 2

        self.CommRange = 150

        self.delta_t = 0.1
        self.AutoCarID = 'Car'
        self.PrevSpeed = 0
        self.PrevVehDistance = 0
        self.PrevAcceleration = 0
        self.VehicleIds = 0
        self.traFlowNumber = 0
        self.finaTCC = 0
        self.vehicleLength = 5

        self.leftmostLaneSpeed = 16.63

        self.firstOvertakingFlag = 0
        self.firstOvertakingCount = 0
        self.a2 = -1

        self.rearLaneFlag = -1
        self.leftRearLaneID = 'None'
        self.middleRearLaneID = 'None'
        self.rightRearLaneID = 'None'

        self.leftRearDistance = -1
        self.rightRearDistance = -1
        self.middleRearDistance = -1

        self.egoSpeed = -1

        self.leftRearSpeed = -1
        self.rightRearSpeed = -1
        self.middleRearSpeed = -1

        self.fairness = 0
        self.fairnessCount = 0

        self.dfGMM = 'None'
        columns = ['velocity', 'acceleration', 'lane_change', 'delta_velocity', 'deceleration_distance', 'drac',
                   'status_change']
        self.df = pd.read_csv('dataPreprocess/bayesianDataset.csv')[columns]
        self.model = load_model('savedModel/bayes_GMMmodel.pkl')

        self.dfGMM, self.gmm = gaussianProcessing(self.df)

        self.punish = 0

        self.overpassFlag = 0
        self.AutoCarFrontID = 'CarF'
        self.ttc_safe = 3

        self.action_space_vehicle = [-1, 0, 1]
        self.n_actions = len(self.action_space_vehicle)
        self.n_actions = int(self.n_actions)
        self.param_velocity = [0, 30]
        self.n_features = 19

        self.actions = np.array([[0, -1], [1, 0], [2, 1]])



    def reset(self):
        self.TotalReward = 0
        self.numberOfLaneChanges = 0
        self.numberOfOvertakes = 0
        self.currentTrackingVehId = 'None'
        self.overpassFlag = 0
        self.firstOvertakingFlag = 0
        self.firstOvertakingCount = 0
        self.fairness = 0
        self.fairnessCount = 0
        self.leftRearLaneID = 'None'
        self.middleRearLaneID = 'None'
        self.rightRearLaneID = 'None'
        self.leftRearSpeed = -1
        self.rightRearSpeed = -1
        self.middleRearSpeed = -1
        self.leftRearDistance = -1
        self.rightRearDistance = -1
        self.middleRearDistance = -1
        self.egoSpeed = -1
        self.punish = 0

        traci.close()

        columns = ['velocity', 'acceleration', 'lane_change', 'delta_velocity', 'deceleration_distance', 'drac',
                   'status_change']
        df = pd.read_csv('dataPreprocess/bayesianDataset.csv')[columns]
        self.dfGMM, _ = gaussianProcessing(df)


        sumo_binary = "sumo"
        sumocfg_file = "data/Lane3/StraightRoad.sumocfg"

        sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--delay", "1", "--scale", "1"]
        traci.start(sumo_cmd)

        # traci.load(config_path)
        print('Resetting the layout')
        traci.simulationStep()

        self.VehicleIds = traci.vehicle.getIDList()

        for veh_id in self.VehicleIds:
            traci.vehicle.subscribe(veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION])

        self.state = self._findstate()
        # traci.simulationStep()
        return np.array(self.state)

    def find_action(self, index):
        return self.actions[index][1]

    def step(self, action, action_param):
        x = action
        v_n = (np.tanh(action_param.cpu().numpy()) + 1) * 15
        desired_speed = float(v_n.item())
        Vehicle_Params = traci.vehicle.getAllSubscriptionResults()

        self.punish = 0

        self.PrevSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
        self.PrevVehDistance = Vehicle_Params[self.AutoCarID][tc.VAR_LANEPOSITION]
        self.PrevAcceleration = Vehicle_Params[self.AutoCarID][tc.VAR_ACCELERATION]

        traci.vehicle.setSpeed(self.AutoCarID, desired_speed)

        if x == 1:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            self.firstOvertakingFlag = self.firstOvertakingFlag + 1
            if laneindex != 0:
                traci.vehicle.changeLane(self.AutoCarID, laneindex - 1, 100)
                self.numberOfLaneChanges += 1
                if self.state[3] == -1:
                    self.punish = self.punish - 1
            else:
                self.punish = self.punish - 1
        elif x == -1:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            self.firstOvertakingFlag = self.firstOvertakingFlag + 1
            if laneindex != self.maxLaneNumber:
                traci.vehicle.changeLane(self.AutoCarID, laneindex + 1, 100)
                self.numberOfLaneChanges += 1
                if self.state[3] == -1:
                    self.punish = self.punish - 1
            else:
                self.punish = self.punish - 1

        laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
        if laneindex == self.maxLaneNumber:
            vehicleSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
            if vehicleSpeed < self.leftmostLaneSpeed:
                self.punish = self.punish - (self.leftmostLaneSpeed - vehicleSpeed)

        traci.simulationStep()

        beforeLeftRearLaneID = self.leftRearLaneID
        beforeMiddleRearLaneID = self.middleRearLaneID
        beforeRightRearLaneID = self.rightRearLaneID

        beforeLeftRearSpeed = self.leftRearSpeed
        beforeMiddleRearSpeed = self.middleRearSpeed
        beforeRightRearSpeed = self.rightRearSpeed

        beforeLeftRearDistance = self.leftRearDistance
        beforeMiddleRearDistance = self.middleRearDistance
        beforeRightRearDistance = self.rightRearDistance

        egoSpeed = self.egoSpeed

        self.state = self._findstate()

        self.checkRearISLaneChane(x, beforeLeftRearLaneID, beforeMiddleRearLaneID, beforeRightRearLaneID, beforeLeftRearDistance,
                                  beforeMiddleRearDistance, beforeRightRearDistance, beforeLeftRearSpeed,
                                  beforeMiddleRearSpeed, beforeRightRearSpeed, egoSpeed)

        if self.rearLaneFlag == 0:
            self.evaluationIndicators_Fairness(x, beforeLeftRearSpeed, beforeMiddleRearSpeed, beforeRightRearSpeed)

        self.end = self.is_overtake_complete(self.state)

        reward = self.updateReward(action, self.state)

        return self.state, reward, self.end

    def evaluationIndicators_Fairness(self, x, beforeLeftRearSpeed, beforeMiddleRearSpeed, beforeRightRearSpeed):
        if x == -1:
            self.fairness = self.fairness + self.state[8] - beforeLeftRearSpeed
            self.fairnessCount = self.fairnessCount + 1
        elif x == 1:
            self.fairness = self.fairness + self.state[12] - beforeRightRearSpeed
            self.fairnessCount = self.fairnessCount + 1
        else:
            self.fairness = self.fairness + self.state[4] - beforeMiddleRearSpeed
            self.fairnessCount = self.fairnessCount + 1


    def checkRearISLaneChane(self, x, beforeLeftRearLaneID, beforeMiddleRearLaneID, beforeRightRearLaneID, beforeLeftRearDistance,
                                  beforeMiddleRearDistance, beforeRightRearDistance, beforeLeftRearSpeed,
                                  beforeMiddleRearSpeed, beforeRightRearSpeed, egoSpeed):
        checkLeftFlag = 0
        checkMiddleFlag = 0
        checkRightFlag = 0
        if beforeLeftRearLaneID != 'None' and (beforeLeftRearDistance < self.ttc_safe * abs(beforeLeftRearSpeed - egoSpeed)):
            checkLeftFlag = 1
        if beforeMiddleRearLaneID != 'None' and (beforeMiddleRearDistance < self.ttc_safe * abs(beforeMiddleRearSpeed - egoSpeed)):
            checkMiddleFlag = 1
        if beforeRightRearLaneID != 'None' and (beforeRightRearDistance < self.ttc_safe * abs(beforeRightRearSpeed - egoSpeed)):
            checkRightFlag = 1

        if x == -1:
            if beforeLeftRearLaneID != self.leftRearLaneID and checkLeftFlag == 1:
                self.rearLaneFlag = 1
            elif beforeLeftRearLaneID == self.leftRearLaneID and checkLeftFlag == 1:
                self.rearLaneFlag = 0
            else:
                self.rearLaneFlag = -1
        elif x == 1:
            if beforeRightRearLaneID != self.rightRearLaneID and beforeRightRearLaneID != 'None' and checkRightFlag == 1:
                self.rearLaneFlag = 1
            elif beforeRightRearLaneID == self.rightRearLaneID and checkLeftFlag == 1:
                self.rearLaneFlag = 0
            else:
                self.rearLaneFlag = -1
        else:
            if beforeMiddleRearLaneID != self.middleRearLaneID and beforeMiddleRearLaneID != 'None' and checkMiddleFlag == 1:
                self.rearLaneFlag = 1
            elif beforeMiddleRearLaneID == self.middleRearLaneID and checkLeftFlag == 1:
                self.rearLaneFlag = 0
            else:
                self.rearLaneFlag = -1

    def close(self):
        traci.close()

    def _findRearVehDistance(self, vehicleparameters):
        parameters = [[0 for x in range(5)] for x in range(len(vehicleparameters))]
        i = 0
        d1 = -1
        d2 = -1
        d3 = -1
        d4 = -1
        d5 = -1
        d6 = -1
        v1 = -1
        v2 = -1
        v3 = -1
        v4 = -1
        v5 = -1
        v6 = -1

        id2 = id4 = id6 = 'None'

        self.a2 = -1
        for VehID in self.VehicleIds:
            parameters[i][0] = VehID
            parameters[i][1] = vehicleparameters[VehID][tc.VAR_LANEPOSITION]  # X position
            parameters[i][2] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # lane Index
            parameters[i][3] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # v
            parameters[i][4] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # a
            i = i + 1

        parameters = sorted(parameters, key=lambda x: x[1])  # Sorted in ascending order based on x distance
        # Find Row with Auto Car
        index = [x for x in parameters if self.AutoCarID in x][0]
        RowIDAuto = parameters.index(index)

        # if there are no vehicles in front
        if RowIDAuto == len(self.VehicleIds) - 1:
            d1 = -1
            v1 = -1
            d3 = -1
            v3 = -1
            d5 = -1
            v5 = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
            # Check if an overtake has happend
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = 'None'
        else:
            # If vehicle is in the lowest lane, then d5,d6,v5,v6 do not exist
            if parameters[RowIDAuto][2] == 0:
                d5 = -1
                v5 = -1
                d6 = -1
                v6 = -1
            # if the vehicle is in the maximum lane index, then d3.d4.v3.v4 do not exist
            elif parameters[RowIDAuto][2] == (self.maxLaneNumber - 1):
                d3 = -1
                v3 = -1
                d4 = -1
                v4 = -1
            # find d1 and v1
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d1 = parameters[index][1] - parameters[RowIDAuto][1]
                    v1 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d1 = -1
                v1 = -1
                self.CurrFrontVehID = 'None'
                self.CurrFrontVehDistance = 150
            # find d3 and v3
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d3 = parameters[index][1] - parameters[RowIDAuto][1]
                    v3 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d3 = -1
                v3 = -1
            # find d5 and v5
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d5 = parameters[index][1] - parameters[RowIDAuto][1]
                    v5 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d5 = -1
                v5 = -1
            # find d2 and v2
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d2 = parameters[RowIDAuto][1] - parameters[index][1]
                    v2 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    id2 = parameters[index][0]
                    self.a2 = vehicleparameters[parameters[index][0]][tc.VAR_ACCELERATION]
                    break
                index -= 1
            if index < 0:
                d2 = -1
                v2 = -1
                self.a2 = -1
            # find d4 and v4
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d4 = parameters[RowIDAuto][1] - parameters[index][1]
                    v4 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    id4 = parameters[index][0]
                    break
                index -= 1
            if index < 0:
                d4 = -1
                v4 = -1
            # find d6 and v6
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d6 = parameters[RowIDAuto][1] - parameters[index][1]
                    v6 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    id6 = parameters[index][0]
                    break
                index -= 1
            if index < 0:
                d6 = -1
                v6 = -1
            # Find if any overtakes has happend
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = parameters[RowIDAuto + 1][0]
        if RowIDAuto == 0:  # This means that there is no car behind
            RearDist = -1
        else:  # There is a car behind return the distance between them
            RearDist = (parameters[RowIDAuto][1] - parameters[RowIDAuto - 1][
                1])
        # Return car in front distance
        if RowIDAuto == len(self.VehicleIds) - 1:
            FrontDist = -1
            # Save the current front vehicle Features
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
        else:
            FrontDist = (parameters[RowIDAuto + 1][1] - parameters[RowIDAuto][
                1])
            # Save the current front vehicle Features
            self.CurrFrontVehID = parameters[RowIDAuto + 1][0]
            self.CurrFrontVehDistance = FrontDist
        # return RearDist, FrontDist
        return d1, v1, d2, v2, id2, d3, v3, d4, v4, id4, d5, v5, d6, v6, id6

    def _findstate(self):
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        # find d1,v1,d2,v2,d3,v3,d4,v4, d5, v5, d6, v6
        d1, v1, d2, v2, id2, d3, v3, d4, v4, id4, d5, v5, d6, v6, id6 = self._findRearVehDistance(VehicleParameters)
        if ((d1 > self.CommRange)):
            d1 = self.maxDistanceFrontVeh
            v1 = -1
        elif d1 < 0:  # if there is no vehicle ahead in L0
            d1 = self.maxDistanceFrontVeh  # as this can be considered as vehicle is far away
        if ((v1 < 0) and (d1 <= self.CommRange)):
            # there is no vehicle ahead in L0 or there is a communication error: # there is no vehicle ahead in L0
            v1 = 0

        if ((d2 > self.CommRange)):
            d2 = self.maxDistanceRearVeh
            v2 = -1
        elif d2 < 0:  # There is no vehicle behind in L0
            d2 = 0  # to avoid negetive reward
        if ((v2 < 0) and (d2 <= self.CommRange)):
            # there is no vehicle behind in L0 or there is a communication error
            v2 = 0
        if ((d3 > self.CommRange)):
            d3 = self.maxDistanceFrontVeh
            v3 = -1
        elif d3 < 0: # no vehicle ahead in L1
            d3 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v3 < 0) and (d3 <= self.CommRange)) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v3 = 0

        if ((d4 > self.CommRange)):
            d4 = self.maxDistanceRearVeh
            v4 = -1
        elif d4 < 0: #There is no vehicle behind in L1
            d4 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v4 < 0) and (d4 <= self.CommRange)) : # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v4 = 0

        if ((d5 > self.CommRange)):
            d5 = self.maxDistanceFrontVeh
            v5 = -1
        elif d5 < 0: # no vehicle ahead in L1
            d5 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v5 < 0) and (d5 <= self.CommRange)) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v5 = 0

        if ((d6 > self.CommRange)):
            d6 = self.maxDistanceRearVeh
            v6 = -1
        elif d6 < 0: #There is no vehicle behind in L1
            d6 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v6 < 0) and (d6 <= self.CommRange)): # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v6 = 0

        va = VehicleParameters[self.AutoCarID][tc.VAR_SPEED]
        da = VehicleParameters[self.AutoCarID][tc.VAR_LANEPOSITION]
        dFront = VehicleParameters[self.AutoCarFrontID][tc.VAR_LANEPOSITION]
        vFront = VehicleParameters[self.AutoCarFrontID][tc.VAR_SPEED]
        # Vehicle acceleration rate
        vacc = (va - self.PrevSpeed)/self.delta_t  # as the time step is 1sec long
        # print("d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6:", d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6)
        if id4 != 'None':
            self.leftRearLaneID = VehicleParameters[id4][tc.VAR_LANE_INDEX]
            self.leftRearSpeed = VehicleParameters[id4][tc.VAR_SPEED]
        else:
            self.leftRearLaneID = 'None'
            self.leftRearSpeed = -1
        if id2 != 'None':
            self.middleRearLaneID = VehicleParameters[id2][tc.VAR_LANE_INDEX]
            self.middleRearSpeed = VehicleParameters[id2][tc.VAR_SPEED]
        else:
            self.middleRearLaneID = 'None'
            self.middleRearSpeed = -1
        if id6 != 'None':
            self.rightRearLaneID = VehicleParameters[id6][tc.VAR_LANE_INDEX]
            self.rightRearSpeed = VehicleParameters[id6][tc.VAR_SPEED]
        else:
            self.rightRearLaneID = 'None'
            self.rightRearSpeed = -1
        self.leftRearDistance = d4
        self.middleRearDistance = d2
        self.rightRearDistance = d6

        self.egoSpeed = va

        return va, da, v1, d1, v2, d2, v3, d3, v4, d4, v5, d5, v6, d6, VehicleParameters[self.AutoCarID][tc.VAR_LANE_INDEX], vacc, self.PrevAcceleration, dFront, vFront

    def is_overtake_complete(self, state):
        delta_v = abs(state[0] - state[18])
        overtake_distance = self.ttc_safe * delta_v
        if (state[1] - state[17] - self.vehicleLength) >= overtake_distance:
            self.overpassFlag = 1
        else:
            self.punish = self.punish - 0.5

        return self.overpassFlag

    def returnFairness(self):
        if self.fairnessCount > 0:
            return self.fairness / self.fairnessCount
        else:
            return 0

    def min_max_normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)


    def TTCCal(self, action, state):
        w_front = 0.5

        if action == -1:
            if state[6] != -1:
                delta_V1 = state[0] - state[6]
                delta_D1 = state[7] - self.vehicleLength
                if delta_V1 <= 0:
                    TCC_front = 10
                else:
                    TCC_front = delta_D1 / delta_V1
            else:
                TCC_front = 10
            if state[8] != -1:
                delta_V2 = state[0] - state[8]
                delta_D2 = state[9] - self.vehicleLength
                if delta_V2 >= 0:
                    TCC_back = 10
                else:
                    TCC_back = delta_D2 / delta_V2
            else:
                TCC_back = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back

        elif action == 1:
            if state[10] != -1:
                delta_V1 = state[0] - state[10]
                delta_D1 = state[11] - self.vehicleLength
                if delta_V1 <= 0:
                    TCC_front = 10
                else:
                    TCC_front = delta_D1 / delta_V1
            else:
                TCC_front = 10
            if state[12] != -1:
                delta_V2 = state[0] - state[12]
                delta_D2 = state[13] - self.vehicleLength
                if delta_V2 >= 0:
                    TCC_back = 10
                else:
                    TCC_back = delta_D2 / delta_V2
            else:
                TCC_back = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back

        else:
            if state[2] != -1:
                delta_V = state[0] - state[2]
                delta_D = state[3] - self.vehicleLength
                if delta_V <= 0:
                    TCC_front = 10
                else:
                    TCC_front = delta_D / delta_V
            else:
                TCC_front = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            TCC_surround = TCC_front

        self.finaTCC = TCC_surround


    def leftTurnAvailabilityCheck(self, state):
        autonomous_vehicle = Vehicle(id='ECV', lane=state[14], position=state[1], velocity=state[0], acceleration=state[15])

        conventional_vehicles = [
            Vehicle(id='NCV1', lane=state[14], position=state[3], velocity=state[2], acceleration=0),
            Vehicle(id='NCV2', lane=state[14], position=state[5], velocity=state[4], acceleration=0),
            Vehicle(id='NCV3', lane=state[14] + 1, position=state[7], velocity=state[6], acceleration=0),
            Vehicle(id='NCV4', lane=state[14] + 1, position=state[9], velocity=state[8], acceleration=0),
            Vehicle(id='NCV5', lane=state[14] - 1, position=state[11], velocity=state[10], acceleration=0),
            Vehicle(id='NCV6', lane=state[14] - 1, position=state[13], velocity=state[12], acceleration=0),
        ]
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

        threshold = 3
        for node in optimal_path:
            if node.state is not None:
                if node.beneETTC < threshold:
                    return 1

        return 0

    def calculateFairness(self, state):
        delta_velocity = state[0]-state[4]
        deceleration_distance = state[5] - self.vehicleLength

        query = {
            'velocity': state[4],
            'acceleration': self.a2,
            'lane_change': self.rearLaneFlag,
            'delta_velocity': delta_velocity,
            'deceleration_distance': deceleration_distance,
            'drac': (delta_velocity) ** 2 / deceleration_distance
        }

        query['velocity_gmm'] = self.gmm[0].predict([[query['velocity']]])[0]
        query['acceleration_gmm'] = self.gmm[1].predict([[query['acceleration']]])[0]
        query['delta_velocity_gmm'] = self.gmm[2].predict([[query['delta_velocity']]])[0]
        query['deceleration_distance_gmm'] = self.gmm[3].predict([[query['deceleration_distance']]])[0]
        query['drac_gmm'] = self.gmm[4].predict([[query['drac']]])[0]

        query.pop('velocity')
        query.pop('acceleration')
        query.pop('delta_velocity')
        query.pop('deceleration_distance')
        query.pop('drac')

        fairIndex = prediction(self.dfGMM, self.model, query)
        return fairIndex

    def updateReward(self, action, state):
        a_max = 5
        ttc_max = 10

        w_f = 1
        w_p = 1
        w_e = 1
        w_s = 1
        w_t = 1

        # reward related to efficiency
        r_ca = abs(state[15]) / a_max
        r_cj = abs(state[15] - state[16]) / (a_max * self.delta_t)
        r_c = - (r_ca + r_cj) / 2

        r_v = self.min_max_normalize(state[0], 0, 30)

        r_effi_all = r_c + r_v

        self.TTCCal(action, state)

        if self.finaTCC < self.ttc_safe:
            r_safe = -1 * self.min_max_normalize(self.finaTCC, 0, self.ttc_safe)
        else:
            r_safe = self.finaTCC / ttc_max

        if self.rearLaneFlag == 0:
            r_fairness = (1 - self.calculateFairness(state)) / 100
        elif self.rearLaneFlag == -1:
            r_fairness = 1
        else:
            r_fairness = 0
            self.fairness = self.fairness - 1
            self.fairnessCount = self.fairnessCount + 1


        r_punish = self.punish

        r_trafficRule = 0
        if self.firstOvertakingFlag == 1 and self.firstOvertakingCount == 0:
            if state[14] == self.maxLaneNumber:
                r_trafficRule = 0
            else:
                availabilityFlag = self.leftTurnAvailabilityCheck(state)
                self.firstOvertakingCount = self.firstOvertakingCount + 1
                if availabilityFlag == 1:
                    if action == -1:
                        r_trafficRule = -1
                    elif action == 1:
                        r_trafficRule = 1
                else:
                    if action == -1:
                        r_trafficRule = 1
                    elif action == 1:
                        r_trafficRule = -1

        # total reward
        r_total = w_e * r_effi_all + w_s * r_safe + w_f * r_fairness + w_p * r_punish + w_t * r_trafficRule

        return r_total

    def getFinaTCC(self):

        return self.finaTCC



