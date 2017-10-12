import vgg
import util
import loss


class Artist:
    def __init__(self, learning_rate, ALPHA, BETA, content, style, vgg):
        self.learning_rate = learning_rate

        self.ALPHA = ALPHA
        self.BETA = BETA
        self.content = content
        self.style = style

        self.vgg = vgg

