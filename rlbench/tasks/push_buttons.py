from typing import List
import itertools
import math
import numpy as np
import random
import re
from scipy.spatial.transform import Rotation as R

from pyrep.const import ObjectType
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from rlbench.backend.task import Task
from rlbench.backend.observation import Observation
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import Condition


MAX_TARGET_BUTTONS = 3
MAX_VARIATIONS = 50

MAX_ITERATIONS = 4

# button top plate and wrapper will be be red before task completion
# and be changed to cyan upon success of task, so colors list used to randomly vary colors of
# base block will be redefined, excluding red and green
colors = [
    ("maroon", (0.5, 0.0, 0.0)),
    ("green", (0.0, 0.5, 0.0)),
    ("blue", (0.0, 0.0, 1.0)),
    ("navy", (0.0, 0.0, 0.5)),
    ("yellow", (1.0, 1.0, 0.0)),
    ("cyan", (0.0, 1.0, 1.0)),
    ("magenta", (1.0, 0.0, 1.0)),
    ("silver", (0.75, 0.75, 0.75)),
    ("gray", (0.5, 0.5, 0.5)),
    ("orange", (1.0, 0.5, 0.0)),
    ("olive", (0.5, 0.5, 0.0)),
    ("purple", (0.5, 0.0, 0.5)),
    ("teal", (0, 0.5, 0.5)),
    ("azure", (0.0, 0.5, 1.0)),
    ("violet", (0.5, 0.0, 1.0)),
    ("rose", (1.0, 0.0, 0.5)),
    ("black", (0.0, 0.0, 0.0)),
    ("white", (1.0, 1.0, 1.0)),
]

color_permutations = list(itertools.permutations(colors, 3))


def print_permutations(color_permutations):
    # pretty printing color_permutations for debug
    print("num permutations: ", str(len(color_permutations)))
    print("color_permutations:\n")
    for i in range(len(color_permutations)):
        print(str(color_permutations[i]))
        if (i + 1) % 16 == 0:
            print("")


class PushButtons(Task):
    def init_task(self) -> None:
        self.color_variation_index = 0
        self.target_buttons = [Shape("push_buttons_target%d" % i) for i in range(3)]
        self.target_topPlates = [
            Shape("target_button_topPlate%d" % i) for i in range(3)
        ]
        self.target_wraps = [Shape("target_button_wrap%d" % i) for i in range(3)]
        self.boundaries = Shape("push_buttons_boundary")
        self.min_position, self.max_position = self.get_bounding_box()
        self.it_condition = ItCondition()

    def init_episode(self, index: int) -> List[str]:
        for tp in self.target_topPlates:
            tp.set_color([1.0, 0.0, 0.0])
        for w in self.target_wraps:
            w.set_color([1.0, 0.0, 0.0])
        # For each color permutation, we want to have 1, 2 or 3 buttons pushed
        color_index = int(index / MAX_TARGET_BUTTONS)
        self.buttons_to_push = 1 + index % MAX_TARGET_BUTTONS
        button_colors = color_permutations[color_index]

        self.color_names = []
        self.color_rgbs = []
        self.chosen_colors = []
        i = 0
        for b in self.target_buttons:
            color_name, color_rgb = button_colors[i]
            self.color_names.append(color_name)
            self.color_rgbs.append(color_rgb)
            self.chosen_colors.append((color_name, color_rgb))
            b.set_color(color_rgb)
            i += 1

        rtn0 = "push the %s button" % self.color_names[0]
        rtn1 = "press the %s button" % self.color_names[0]
        rtn2 = "push down the button with the %s base" % self.color_names[0]
        for i in range(self.buttons_to_push):
            if i == 0:
                continue
            else:
                rtn0 += ", then push the %s button" % self.color_names[i]
                rtn1 += ", then press the %s button" % self.color_names[i]
                rtn2 += ", then the %s one" % self.color_names[i]

        b = SpawnBoundary([self.boundaries])
        for button in self.target_buttons:
            b.sample(button, min_distance=0.1)

        num_non_targets = 3 - self.buttons_to_push
        spare_colors = list(
            set(colors)
            - set([self.chosen_colors[i] for i in range(self.buttons_to_push)])
        )

        spare_color_rgbs = []
        for i in range(len(spare_colors)):
            _, rgb = spare_colors[i]
            spare_color_rgbs.append(rgb)

        color_choice_indexes = np.random.choice(
            range(len(spare_colors)), size=num_non_targets, replace=False
        )
        non_target_index = 0
        for i, button in enumerate(self.target_buttons):
            if i in range(self.buttons_to_push):
                pass
            else:
                _, rgb = spare_colors[color_choice_indexes[non_target_index]]
                button.set_color(rgb)
                non_target_index += 1

        self.register_waypoints_should_repeat(self._repeat)
        self.register_success_conditions([self.it_condition])
        self.it_condition.reset()

        return [rtn0, rtn1, rtn2]

    def variation_count(self) -> int:
        return np.minimum(len(color_permutations) * MAX_TARGET_BUTTONS, MAX_VARIATIONS)

    def step(self) -> None:
        return None

    def _repeat(self):
        self.it_condition.iterate()
        if self.it_condition.condition_met()[0]:
            return False

        iteration = self.it_condition.iterations
        original_waypoints_num = len(self.original_waypoints)
        if not self.push_on_start:
            # Use random waypoints until min_it
            min_it = MAX_ITERATIONS - original_waypoints_num
            if iteration >= min_it:
                self.update_waypoint(iteration - min_it)
            else:
                self.update_waypoint(None)
        else:
            # Use original waypoints until max_it
            max_it = original_waypoints_num
            if iteration >= max_it:
                self.update_waypoint(None)
            else:
                self.update_waypoint(iteration)

        return True

    # Trick to obtain the waypoints after the task is randomized in the workspace.
    def validate(self):
        waypoints = self.get_all_waypoints()
        self.waypoint = waypoints[0]
        self.original_waypoints = []
        for idx, waypoint in enumerate(waypoints):
            position = waypoint.get_position()
            orientation = waypoint.get_orientation()
            self.original_waypoints.append((position, orientation))
            if idx > 0:
                waypoint.set_name(f"old_waypoint{idx}")

        self.push_on_start = random.choice([True, False])
        print("PUSH ON START:", self.push_on_start)
        if not self.push_on_start:
            # Update waypoints to random pose on start.
            self.update_waypoint(original_waypoint=None)
        super().validate()

    def cleanup(self) -> None:
        old_waypoints = self.get_all_waypoints("old_waypoint", 1)
        for old_waypoint in old_waypoints:
            name = old_waypoint.get_name()
            numbers = re.findall(r"\d+", name)
            waypoint_name = "waypoint" + "".join(numbers)
            old_waypoint.set_name(waypoint_name)

    def update_waypoint(self, original_waypoint=None):
        self.pyrep.step()
        if original_waypoint is not None:
            position, orientation = self.original_waypoints[original_waypoint]
        else:
            position, orientation = self.get_random_pose(
                self.min_position, self.max_position
            )
        self.waypoint.set_position(position)
        self.waypoint.set_orientation(orientation)

    def calculate_limits(self, position, bounding_box):
        x_min = position[0] + bounding_box[0]
        x_max = position[0] + bounding_box[1]

        y_min = position[1] + bounding_box[2]
        y_max = position[1] + bounding_box[3]

        z_min = position[2] + bounding_box[4]
        z_max = position[2] + bounding_box[5]

        return [x_min, y_min, z_min], [x_max, y_max, z_max]

    def get_bounding_box(self):
        table_name = "diningTable"
        if Object.exists(table_name):
            table_obj = Object.get_object(table_name)
            table_position = table_obj.get_position()
            table_bounding_box = table_obj.get_bounding_box()
            min_position, max_position = self.calculate_limits(
                table_position, table_bounding_box
            )
            current_robot_position = self.robot.arm.get_tip().get_position()
            min_position[2] = max_position[2] + 0.01  # min_z as height of the table.
            max_position[2] = current_robot_position[
                2
            ]  # max_z as starting pose of the gripper pos.

        return min_position, max_position

    def get_random_pose(self, min_position, max_position):
        while (
            True
        ):  # Infinite loop might be aggresive but we can't continue without waypoints.
            x = np.random.uniform(min_position[0], max_position[0])
            y = np.random.uniform(min_position[1], max_position[1])
            z = np.random.uniform(min_position[2], max_position[2])
            position = [x, y, z]

            a, b, c = random.random(), random.random(), random.random()

            s = math.sqrt(1 - a)
            t = math.sqrt(a)

            q1 = math.sin(2 * math.pi * b) * s
            q2 = math.cos(2 * math.pi * b) * s
            q3 = math.sin(2 * math.pi * c) * t
            q4 = math.cos(2 * math.pi * c) * t

            quaternion = [q1, q2, q3, q4]
            r = R.from_quat(quaternion)
            euler = r.as_euler("xyz", degrees=False)  # [roll, pitch, yaw]
            try:
                self.robot.arm.get_linear_path(position=position, euler=euler)
                return position, euler
            except Exception:
                continue

    def get_all_waypoints(self, waypoint_initial="waypoint", initial_index=0):
        def get_name(i):
            return f"{waypoint_initial}{i}"

        i = initial_index
        name = get_name(i)
        waypoint_objects: List[Object] = []
        while Object.exists(name):
            ob_type = Object.get_object_type(name)

            if ob_type == ObjectType.DUMMY:
                obj = Object.get_object(name)
                waypoint_objects.append(obj)
            i += 1
            name = get_name(i)
        return waypoint_objects

    def decorate_observation(self, observation: Observation) -> Observation:
        waypoints = self.get_all_waypoints()
        current_waypoint = np.concatenate(
            (waypoints[0].get_position(), waypoints[0].get_orientation())
        )
        observation.waypoint = current_waypoint
        return observation


class ItCondition(Condition):
    def __init__(self):
        self.iterations = 0

    def iterate(self):
        self.iterations += 1

    def condition_met(self):
        met = self.iterations >= MAX_ITERATIONS
        return met, False

    def reset(self):
        self.iterations = 0
