from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import pickle

class graphic(object):

    def init_func(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # axis dimensions
        self.ax.set_xlim([0, 10])
        self.ax.set_ylim([0, 10])

        # uav_list
        self.uavs = []
        self.di = list()  # np.zeros((N,N))
        # generate meshgrid
        self.x = np.array([x for _ in range(0, 10) for x in range(0, 11)])
        self.y = np.array([y for y in range(0, 10) for _ in range(0, 11)])

        # pack x,y in one list
        self.x_y = np.array(list(zip(self.x, self.y)))


# plot scatter plot of mesh grid

# plot the closest of the button click point
    def plot_closest_pts(self,x, y):
        ind = distance.cdist([[x, y]], self.x_y, 'euclidean').argmin()

        # prevent duplicate points
        self.uavs.append(tuple(self.x_y[ind]))
        if len(self.uavs) != len(set(self.uavs)):
            self.uavs.pop(len(self.uavs) - 1)

        plt.plot(self.x_y[ind][0], self.x_y[ind][1], color='red', marker='o', markersize=9)


    # function to be followed on every button click
    def onclick(self,event):
        # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %          (event.button, event.x, event.y, event.xdata, event.ydata))

        self.plot_closest_pts(event.xdata, event.ydata)
        # plt.plot(event.xdata, event.ydata, color='green', marker='o', markersize=9)

        # unzip x_y
        temp_x, temp_y = zip(*self.uavs)
        plt.plot(temp_x, temp_y)

        print("uavs", self.uavs)
        # distance matrix

        print("Distance matrix:", self.di)
        # print("n")
        self.fig.canvas.draw()


    def display(self):
        self.init_func()
        plt.scatter(self.x, self.y)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show()
        d_matrix = distance.cdist(self.uavs, self.uavs, 'euclidean')
    
        print("Reached here")
        if plt.close() == True:
            print("OKAY")
        plt.close()
        return d_matrix

