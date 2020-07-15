import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import QTimer, Qt
from pong.game_env import GameEnv
from pong.learn_game import GameLearner
import numpy as np

class GameGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.game_env = GameEnv()

        self.initUI()

        self.timer = QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start()

        self.game_player = GameLearner()
        self.game_player.model.load_weights("pong/models/trial-120.model.w")

    def initUI(self):
        self.setWindowTitle("Game GUI")
        self.resize(self.game_env.env_width, self.game_env.env_height)
        self.show()

    def paintEvent(self, e):
        p = QPainter()
        p.begin(self)

        # background.
        p.setBrush(QColor(0, 0, 0))
        p.drawRect(0, 0, self.game_env.env_width, self.game_env.env_height)

        # ball
        p.setBrush(QColor(100, 100, 100))
        ball_x = int((self.game_env.state[0] + 0.5) * self.game_env.env_width)
        ball_y = int(self.game_env.state[1] * self.game_env.env_height)
        ball_size = self.game_env.ball_size
        p.drawEllipse(ball_x - ball_size/2, ball_y - ball_size/2, ball_size, ball_size)

        # bar
        bar_width = self.game_env.bar_width
        bar_height = self.game_env.bar_height

        bar_x = (self.game_env.state[4] + 0.5) * self.game_env.env_width
        bar_y = self.game_env.env_height - bar_height

        p.setBrush(QColor(0, 0, 255))
        p.drawRect(bar_x - (bar_width / 2), bar_y, bar_width, bar_height)

        # score
        p.setPen(QColor(0, 255, 0))
        p.drawText(0, 15, 50, 30, Qt.AlignRight, str(self.game_env.score))

        p.end()

    def timerEvent(self):

        if self.game_env.game_over == 0:

            # if auto mode.
            state = np.copy(self.game_env.state)
            state = np.reshape(state, (1, len(state)))
            y = self.game_player.model.predict(state)[0]
            action = np.argmax(y)
            self.game_env.act(action)

            self.game_env.step()

            print("action: {0}, reward: {1}".format(action, y[action]))

        self.repaint()

    def keyPressEvent(self, e):

        if e.key() == Qt.Key_Left:
            self.game_env.act(1)

        elif e.key() == Qt.Key_Right:
            self.game_env.act(2)

        elif e.key() == Qt.Key_Space:
            self.game_env.init()

if __name__ == "__main__":
    print("Game GUI")

    app = QApplication(sys.argv)
    gui = GameGUI()

    ret = app.exec_()
    sys.exit(ret)


