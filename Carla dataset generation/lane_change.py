
import random
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      KeepVelocity,
                                                                      Idle,
                                                                      ActorDestroy)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario


class LaneChangeR(BasicScenario):

    timeout = 120            # Timeout of scenario in seconds
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=30):
        
        #self.world = world # world definiert die Umgebung (nicht die map)
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        
        # velocity in m/s not in km/h
        self._slow_vehicle_velocity = 16 # langsames Fahrzeug (60 km/h)
        self._fast_vehicle_velocity = 25 # schnelles Fahrzeug (90 km/h)
        self._change_lane_velocity = 0
        self._max_brake = 1

        self.direction = 'left'  # direction of lane change
        self.lane_check = 'true'  # check whether a lane change is possible

        # Call constructor of BasicScenario
        super(LaneChangeR, self).__init__(
          "LaneChange",
          ego_vehicles,
          config,
          world,
          debug_mode,
          criteria_enable=criteria_enable)

    def _initialize_actors(self, config):

        # fast vehicle
        fast_vehicle_transform = config.other_actors[0].transform # erstellen des Fahrzeuges mit dem Modell und Postionsdaten aus Config
        fast_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, fast_vehicle_transform)
        fast_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(fast_vehicle) # hinzufügen zur Liste der Aktoren
        self._fast_actor_transform = fast_vehicle_transform #Fahrzeuge werden in ActorTransformsetter in create_behavior gesetzt

        # slow vehicle
        slow_vehicle_transform = config.other_actors[1].transform
        slow_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[1].model, slow_vehicle_transform)
        slow_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(slow_vehicle) 
        self._slow_actor_transform = slow_vehicle_transform 
        
    def _create_behavior(self):

        # fast vehicle
        sequence_fast_vehicle = py_trees.composites.Sequence("fastVehicle")
            # 1. spawn fast vehicle
        spawn_fast_vehicle = ActorTransformSetter(self.other_actors[0], self._fast_actor_transform)
        sequence_fast_vehicle.add_child(spawn_fast_vehicle)
            # 2.driving towards slow vehicle with defined velocity
        driving_fast_vehicle = py_trees.composites.Parallel("DrivingTowardsSlowVehicle",
                                                  policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity_fast_vehicle = WaypointFollower(self.other_actors[0], self._fast_vehicle_velocity, avoid_collision=True)
        keep_distance_fast_vehicle= InTriggerDistanceToVehicle(self.other_actors[1], self.other_actors[0],10) # lane change should start 10m behind slow vehicle
        driving_fast_vehicle.add_child(keep_velocity_fast_vehicle)
        driving_fast_vehicle.add_child(keep_distance_fast_vehicle)
        sequence_fast_vehicle.add_child(driving_fast_vehicle)
            # 3. lane change maneuver 
        change_lane_and_driving = py_trees.composites.Sequence("LaneChangeAndContinueDrivingFast")
        continue_driving=WaypointFollower(self.other_actors[0], self._fast_vehicle_velocity)
        change_lane = LaneChange(self.other_actors[0], distance_other_lane=10) # distance other lange = zu fahrende Strecke auf neuer Spur bis nächster Sequenzpunkt startet
        change_lane_and_driving.add_child(change_lane)
        change_lane_and_driving.add_child(continue_driving)
        sequence_fast_vehicle.add_child(change_lane_and_driving)

        # slow vehicle
        sequence_slow_vehicle = py_trees.composites.Sequence("slowVehicle")
            # 1. spawn slow vehicle 
        spawn_slow_vehicle = ActorTransformSetter(self.other_actors[1], self._slow_actor_transform)
        sequence_slow_vehicle.add_child(spawn_slow_vehicle)
            # 2. driving with defined velocity
        drive_slow_vehicle = py_trees.composites.Sequence("DrivingSlowVehicle")
        drive_slow_vehicle.add_child(WaypointFollower(self.other_actors[1], self._slow_vehicle_velocity, avoid_collision=False))
        sequence_slow_vehicle.add_child(drive_slow_vehicle)
        
        # complete tree    
        root = py_trees.composites.Parallel("Parallel Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        root.add_child(sequence_slow_vehicle)
        root.add_child(sequence_fast_vehicle)
       
        return root

    def _create_test_criteria(self):
        """
        Setup the evaluation criteria 
        """
        criteria = []

        collision_criteria  = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criteria)

        return criteria
    def __del__(self):
        self.remove_all_actors()
