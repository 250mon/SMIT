import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb


class HistEqual():
    def __init__(self, filename):
        # load a image, e.g (232, 230, 3)
        self.in_image = cv2.imread(filename)
        # convert BGR to YCrCb, e.g  (232, 230, 3)
        self.in_YCrCb = self._getYCbCr(self.in_image)
        # get Y, e.g. (232, 230)
        self.in_Y = None
        self.outs = []

        # For plotting histograms showing images
        self.original_img_shown = False
        self.histos = []

    # convert BGR to YCbCr
    def _getYCbCr(self, image):
        YCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        return YCbCr

    # convert YCbCr to BGR
    def _getBGR(self, image):
        bgr = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        return bgr

    # return a copy of Y matrix
    def _get_Y(self):
        if type(self.in_Y) == type(None):
            self.in_Y = self.in_YCrCb[:, :, 0]
            # save the histogram of the original image
            self.histos.append(self._get_histogram_2D(self.in_Y))

        # return its copy in order not to corrupt the original cvt
        return self.in_Y.copy()

    # in_array should be unsigned int
    def _get_histogram(self, in_array):
        hist = np.bincount(in_array) / len(in_array)
        return hist

    # 2d matrix histogram
    def _get_histogram_2D(self, in_matrix):
        temp_array = np.concatenate(in_matrix, axis=0)
        hist = self._get_histogram(temp_array)
        return hist

    # Save the histogram for later plotting
    def _save_histogram(self, in_array):
        self.histos.append(in_array)

    # in_matrix should be unsigned int
    def _get_tfvec(self, in_matrix):
        temp_array = np.concatenate(in_matrix, axis=0)
        # obtain histogram
        hist = self._get_histogram(temp_array)
        # histogram cumulative prob
        out_array = 255. * np.cumsum(hist)
        out_array = out_array.astype(int)
        # pdb.set_trace()

        return out_array

    def _cv2conv(self, image, kernel):
        """ Convolution using opencv filter2D function """
        # kernel flipping (left-right and up-down)
        flip_ker = np.flipud(np.fliplr(kernel))
        # 2D filtering
        # data , offset (output position: -1 means center),
        # flipped-kernel, border-processing
        return cv2.filter2D(image, -1, flip_ker, borderType=cv2.BORDER_REFLECT_101)

    # histogram equalization transform
    def equalize_hist(self, in_matrix=None):
        if in_matrix is None:
            in_matrix = self._get_Y()
        tf_vec = self._get_tfvec(in_matrix)
        out_matrix = np.take(tf_vec, in_matrix)

        # save histogram for later plotting
        hist_out = self._get_histogram_2D(out_matrix)
        self._save_histogram(hist_out)

        # save output images
        out_YCrCb = self.in_YCrCb.copy()
        out_YCrCb[:, :, 0] = out_matrix
        self.outs.append(out_YCrCb)

    # if x >= avg(neighbor), replace x with 1, or 0
    # return a matrix after adding 1 or 0 to the original value
    def lambda2_avg(self, in_matrix=None):
        if in_matrix is None:
            in_matrix = self._get_Y()
        # define a kernel to take a average of the neighborhood
        kernel = np.array([[1/8, 1/8, 1/8],
                           [1/8,  0., 1/8],
                           [1/8, 1/8, 1/8]])
        # make a matrix of neighborhood avg
        avg_neighbor = self._cv2conv(in_matrix, kernel)
        # sorting bins by avg_neighbor
        out_matrix = in_matrix * 2.0 + np.where(in_matrix >= avg_neighbor, 1., 0.)
        out_matrix = out_matrix.astype(int)
        return out_matrix

    def lambda2_inverted_avg(self, in_matrix=None):
        if in_matrix is None:
            in_matrix = self._get_Y()
        # define a kernel to take a average of the neighborhood
        kernel = np.array([[1/8, 1/8, 1/8],
                           [1/8,  0., 1/8],
                           [1/8, 1/8, 1/8]])
        # make a matrix of neighborhood avg
        avg_neighbor = self._cv2conv(in_matrix, kernel)
        # inverted avg (-255 ~ 255)
        alpha_m = in_matrix - avg_neighbor
        # sorting bins by alpha_m
        out_matrix = in_matrix * 511.0 + alpha_m
        out_matrix = out_matrix.astype(int)
        return out_matrix

    def lambda2_voting(self, in_matrix=None):
        if in_matrix is None:
            in_matrix = self._get_Y()
        # define a kernel for voting
        kernels = []
        for m in range(3):
            for n in range(3):
                if m != 1 and n != 1:
                    kernel = np.zeros((3, 3))
                    kernel[m, n] = -1.0
                    kernel[1, 1] = 1.0
                    kernels.append(kernel)
        # sorting by voting (0 ~ 8)
        out_matrix = in_matrix * 9.0
        for kernel in kernels:
            voting = self._cv2conv(in_matrix, kernel)
            voting = np.where(voting > 0, 1, 0)
            out_matrix = out_matrix + voting
        out_matrix = out_matrix.astype(int)
        return out_matrix

    def plot_histogram(self):
        histos_size = len(self.histos)
        if histos_size == 0:
            return

        fig, ax = plt.subplots(histos_size)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(histos_size):
            ax[i].plot(self.histos[i])
            ax[i].set_title(f"Histogram {i}")
        plt.show()

    def show_img(self):
        cv2.imshow('original', self.in_image)
        for i in range(len(self.outs)):
            cv2.imshow(f'transformed {i}', self._getBGR(self.outs[i]))


if __name__ == '__main__':
    hist_equal = HistEqual("../Images/0011_Low-contrast-image.png")

    # # Classical HE
    hist_equal.equalize_hist()

    # General sorting(neighborhood avg) and HE
    temp_matrix = hist_equal.lambda2_avg()
    hist_equal.equalize_hist(temp_matrix)

    # Sorting(inverted avg) and HE
    temp_matrix = hist_equal.lambda2_inverted_avg()
    hist_equal.equalize_hist(temp_matrix)

    # Sorting(voting) and HE
    temp_matrix = hist_equal.lambda2_voting()
    hist_equal.equalize_hist(temp_matrix)

    # display images and histograms
    hist_equal.show_img()
    hist_equal.plot_histogram()

    print("press any key to proceed or press 'q' to quit")
    key = cv2.waitKey(0)
