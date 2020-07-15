import random

class GameEnv:
    def __init__(self):
        self.env_width = 300
        self.env_height = 200

        self.ball_size = 10
        self.bar_width = 50
        self.bar_height = 5
        self.v_noise = 0.2

        # state: ball x, ball y, ball vx, ball vy, bar x.
        nvx = random.uniform(-self.v_noise, self.v_noise)
        self.state = [0, (self.ball_size * 0.5) / self.env_height, nvx, 1, 0]

        self.max_steps = self.env_height * 2 * 50

        self.steps = 0
        self.score = 0

        self.num_actions = 3             # idle, move left, right

        self.game_clear = 0
        self.game_over = 0
        self.reward = 0

    def init(self):
        nvx = random.uniform(-self.v_noise, self.v_noise)

        self.state = [0, (self.ball_size * 0.5) / self.env_height, nvx, 1, 0]

        self.steps = 0
        self.score = 0
        self.game_over = 0

    def set_state(self, state):
        self.state = state

    def step(self):
        ball_x = (self.state[0] + 0.5) * self.env_width
        ball_y = self.state[1] * self.env_height
        ball_vx = self.state[2]
        ball_vy = self.state[3]
        bar_x = (self.state[4] + 0.5) * self.env_width

        bar_left = bar_x - (self.bar_width / 2)
        bar_right = bar_x + (self.bar_width / 2)
        bar_top = self.env_height - self.bar_height

        # move ball
        ball_x = ball_x + ball_vx
        ball_y = ball_y + ball_vy

        nvx = random.uniform(-self.v_noise, self.v_noise)
        self.reward = 0
        self.terminal = False

        # collision test.
        # left wall
        if ball_x < (self.ball_size/2):
            # change motion vector.
            ball_vx = -ball_vx + nvx

        # right wall
        elif ball_x >= (self.env_width - (self.ball_size/2)):
            # change motion vector.
            ball_vx = -ball_vx + nvx

        # ceiling
        elif ball_y <= (self.ball_size/2):
            ball_vy = -ball_vy
            ball_vx = ball_vx + nvx

        # bar
        elif bar_left <= ball_x <= bar_right and ball_y > (bar_top - self.ball_size * 0.5):
            ball_vy = -ball_vy
            ball_vx = ball_vx + nvx

            ball_y = min(ball_y, bar_top - self.ball_size * 0.5)

            self.score += 1
            self.reward = 10
            self.terminal = True

        # bottom
        elif ball_y > self.env_height:
            self.game_over = 1
            self.reward = -100.0
            self.terminal = True

        ball_vx = max(-1, min(ball_vx, 1))

        ball_x = max(ball_x, self.ball_size / 2)
        ball_y = max(ball_y, self.ball_size / 2)

        ball_x = min(ball_x, self.env_width - self.ball_size / 2)

        self.state[0] = (ball_x / self.env_width) - 0.5
        self.state[1] = ball_y / self.env_height
        self.state[2] = ball_vx
        self.state[3] = ball_vy
        self.state[4] = (bar_x / self.env_width) - 0.5

        self.steps += 1

        if self.steps >= self.max_steps:
            self.game_clear = 1

    def act(self, action):

        if action == 1:
            bar_vx = -2
        elif action == 2:
            bar_vx = 2
        else:
            bar_vx = 0

        bar_x = (self.state[4] + 0.5) * self.env_width

        bar_x = bar_x + bar_vx

        if bar_x < (self.bar_width/2):
            bar_x = self.bar_width / 2
        elif bar_x > (self.env_width - self.bar_width/2):
            bar_x = (self.env_width - self.bar_width/2)

        self.state[4] = (bar_x / self.env_width) - 0.5



