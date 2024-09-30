import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame.locals import *
import sys
import math


L = 2.56  # [m]
Lr = L / 2.0  # [m]
Lf = L - Lr
Cf = 1600.0 * 2.0  # N/rad
Cr = 1700.0 * 2.0  # N/rad
Iz = 2250.0  # kg/m2
m = 1500.0  # kg

class DynamicBicycle:

    def __init__(self, x=0.0, y=0.0, theta=0.0, vx=0.01, vy=0.0, omega=0.0):
        self.xc = x
        self.yc = y
        self.theta = theta
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.delta = 0.0
        self.L = L
        self.lr = Lr
        
    def step(self, a, d_delta, dt=0.2):
        if self.vx < 0.1:
            self.stepKinematic(a, d_delta)
        else:
            self.stepDynamic(a, d_delta)
        
    def stepKinematic(self, a, d_delta, dt=0.2):
        dx = self.vx*np.cos(self.theta) * dt 
        dy = self.vx*np.sin(self.theta) * dt 
        self.xc = self.xc + dx
        self.yc = self.yc + dy
        self.theta = self.theta + self.vx*np.cos(self.omega)*np.tan(self.delta)/self.L * dt
        self.beta = np.arctan(self.lr*self.delta/self.L)
        self.vx += a*dt - self.vx / 25
        self.delta += d_delta
        self.delta = np.clip(self.delta, -0.5, 0.5)
        self.vx = np.clip(self.vx, -20, 20)

    def stepDynamic(self, a, d_delta, dt=0.2):
        self.xc = self.xc + self.vx * math.cos(self.theta) * dt - self.vy * math.sin(self.theta) * dt
        self.yc = self.yc + self.vx * math.sin(self.theta) * dt + self.vy * math.cos(self.theta) * dt
        self.theta = self.theta + self.omega * dt
        Ffy = -Cf * math.atan2(((self.vy + Lf * self.omega) / self.vx - self.delta), 1.0)
        Fry = -Cr * math.atan2((self.vy - Lr * self.omega) / self.vx, 1.0)
        self.vx = self.vx + (a - Ffy * math.sin(self.delta) / m + self.vy * self.omega) * dt - self.vx / 25
        self.vy = self.vy + (Fry / m + Ffy * math.cos(self.delta) / m - self.vx * self.omega) * dt
        self.omega = self.omega + (Ffy * Lf * math.cos(self.delta) - Fry * Lr) / Iz * dt
        self.delta += d_delta
        self.vx = np.clip(self.vx, -20, 20)        
        self.delta = np.clip(self.delta, -0.5, 0.5)


        

car = DynamicBicycle()




D_STEER = 0.2
D_A = 0.75
area = 20

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")



def plot_lmu_sensor_axes(x, y, yaw):
    # Define the length of the IMU axes
    imu_length = 2.0  # Length of the axes

    # IMU X-axis (forward)
    x_end = x + imu_length * np.cos(yaw)
    y_end = y + imu_length * np.sin(yaw)
    plt.plot([x, x_end], [y, y_end], 'k--', label='IMU X-axis')

    # IMU Y-axis (left)
    y_axis_end_x = x + imu_length * np.cos(yaw + np.pi / 2)
    y_axis_end_y = y + imu_length * np.sin(yaw + np.pi / 2)
    plt.plot([x, y_axis_end_x], [y, y_axis_end_y], 'm--', label='IMU Y-axis')

    # IMU Z-axis (upward, represented vertically)
    plt.plot([x, x], [y, y], 'c--', label='IMU Z-axis')




def loop():
    pygame.init()
    window = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()

    physics_dt = 1 / 60.0
    last_time = pygame.time.get_ticks()

    # Set up the plot with a larger size and grid
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 10))

    # Lists to store trajectory points
    trajectory_x = []
    trajectory_y = []
    trajectory_color = []

    while True:
        a, d_steer = 0, 0
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[K_RIGHT]:
            d_steer = -D_STEER
        if keys[K_LEFT]:
            d_steer = D_STEER
        if keys[K_UP]:
            a = D_A
        if keys[K_DOWN]:
            a = -D_A * 2  # Pressing DOWN will reverse

        # Update physics at fixed rate
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - last_time) / 1000.0  # Convert to seconds
        if elapsed_time >= physics_dt:
            car.step(a, d_steer, physics_dt)
            last_time = current_time

            # Append current car position to trajectory lists
            trajectory_x.append(car.xc)
            trajectory_y.append(car.yc)

            # Set trajectory color based on car's velocity (red for reverse, blue for forward)
            trajectory_color.append('r' if car.vx < 0 else 'b')



        # Limit frame rate to 60 FPS
        clock.tick(60)

        # Draw car
        plt.gca().clear()
        plt.xlim([-area, area])
        plt.ylim([-area, area])

        # Plot grid lines
        plt.grid(True)

        # Plot car and LMU sensor axes
        plot_car(car.xc, car.yc, car.theta, car.delta)
        plot_lmu_sensor_axes(car.xc, car.yc, car.theta)

        # Update plot
        plt.draw()
        plt.pause(0.001)

        pygame.display.flip()

loop()