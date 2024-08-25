import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from array import array
from sympy import symbols
from sympy import lambdify
from lmfit import Model, report_fit
import os
import sys
import time

class FitResonances:
    def __init__ (self, freq, real, img):
        self.freq = freq[:-1]
        self.real = real[:-1]
        self.img = img[:-1]
        self.x_inf = real[-1]
        self.y_inf = img[-1]
        self.theta = []
        self.tan_theta = []
        self.l = []
        self.plot = False

    def phase_unwrap(self, theta):
        for i in range (len(theta) - 1):
            diff = np.abs(theta[i] - theta[i+1])
            if np.round(diff, 1) == 3.1:
                theta[i + 1:] = theta[i + 1:] + theta[i] - theta[i + 1]
        return theta
    
    def plot_lorentzian(self, w0, gamma, A, phi, c0=0, c1=0, m0=0, m1=0):
        w = symbols('w')
        X = (w - w0) / ((w - w0) ** 2 + gamma ** 2 / 4) 
        Y = - gamma / 2 / ((w - w0) ** 2 + gamma ** 2 / 4) 
        lam_x = lambdify(w, X, modules=['numpy'])
        lam_y = lambdify(w, Y, modules=['numpy'])
        x_vals = np.linspace(self.freq[0], self.freq[-1], 10000)
        # real = A * (lam_x(x_vals) * np.cos(phi) - lam_y(x_vals) * np.sin(phi)) + c0 + m0 * x_vals
        # img = A * (lam_x(x_vals) * np.sin(phi) + lam_y(x_vals) * np.cos(phi)) + c1 + m1 * x_vals
        real = A * (lam_x(x_vals) * np.cos(phi) - lam_y(x_vals) * np.sin(phi))
        img = A * (lam_x(x_vals) * np.sin(phi) + lam_y(x_vals) * np.cos(phi)) 
        if self.plot:
            plt.figure(figsize=(10,6))
            plt.xlabel("Frequency (MHz)", fontsize = 15)
            plt.ylabel(r"Amplitude", fontsize = 15)
            plt.title("Fit and Data")
            plt.plot(x_vals, real, color = "red", label = "Real")
            plt.plot(x_vals, img, color = "blue", label = "Img")
            plt.ticklabel_format(useOffset=False)
            plt.legend()
            
            plt.xlabel("Frequency (MHz)", fontsize = 15)
            plt.ylabel(r"Amplitude", fontsize = 15)
            plt.scatter(self.freq, self.real, color = "red", s=1)
            plt.scatter(self.freq, self.img, color = "blue", s=1)      
            plt.ticklabel_format(useOffset=False)
            plt.legend()

            plt.figure(figsize=(10,6))
            plt.title("Centered Data")
            plt.xlabel("Frequency (MHz)", fontsize = 15)
            plt.ylabel(r"Amplitude", fontsize = 15)
            plt.scatter(self.freq, self.real - np.average(self.real), color = "red", s=1, label = "Real")
            plt.scatter(self.freq, self.img - np.average(self.img), color = "blue", s=1, label = "Img")    
            plt.ticklabel_format(useOffset=False)
            plt.legend()

            plt.figure(figsize=(10,6))
            plt.title("Centered Fit")
            plt.xlabel("Frequency (MHz)", fontsize = 15)
            plt.ylabel(r"Amplitude", fontsize = 15)
            plt.scatter(x_vals, real - np.average(real), color = "red", s=1, label = "Real")
            plt.scatter(x_vals, img - np.average(img), color = "blue", s=1, label = "Img")    
            plt.ticklabel_format(useOffset=False)
            plt.legend()

            plt.figure()
            plt.plot(self.freq, np.sqrt(self.real ** 2 + self.img ** 2))

            plt.show()
        
        return x_vals, real, img
    """
    Use the legendary Arkady method to generate a guess for the lorentzian parameters.
    https://static-content.springer.com/esm/art%3A10.1038%2Fnature12165/MediaObjects/41586_2013_BFnature12165_MOESM485_ESM.pdf
    """
    def arkady(self):
        for index, _ in enumerate(self.real):
            self.x_shift = self.real[index] - self.x_inf
            self.y_shift = self.img[index] - self.y_inf
            self.theta.append(np.arctan2(self.y_shift, self.x_shift))
            self.l.append(np.sqrt(self.x_shift ** 2 + self.y_shift ** 2))

        self.theta = self.phase_unwrap(self.theta)
        self.tan_theta = np.tan(self.theta)

        start = 100
        end = -100
        
        z = np.polyfit(self.freq[start : end], self.tan_theta[start : end], 1)
        p = np.poly1d(z)
        gamma = 2 / z[0]
        Q = 1 / gamma
        w0 = -1 / z[0] * z[1]
        x_w0 = np.interp(w0, self.freq, self.real)
        y_w0 = np.interp(w0, self.freq, self.img)
        x_center = (x_w0 + self.x_inf) / 2
        y_center = (y_w0 + self.y_inf) / 2
        phi = np.arctan2(y_center, x_center)
        A = np.sqrt((x_w0 - self.x_inf) ** 2 + (y_w0 - self.y_inf) ** 2) * gamma / 2

        if self.plot:
            plt.figure()
            plt.xlabel("Frequency (MHz)", fontsize = 15)
            plt.ylabel(r"tan($\theta$)", fontsize = 15)
            xaxis = np.linspace(self.freq[start], self.freq[end], 1000)
            plt.plot(xaxis, p(xaxis), 
                    color = "red", 
                    label = f"y = {np.round(z[0], 5)}x + {np.round(z[1], 5)} \n Gamma (width) = {np.round(gamma, 5)} \n w_0 = {np.round(w0, 5)}")
            plt.scatter(self.freq[start : end], self.tan_theta[start : end], s=1)
            # plt.scatter(self.freq, self.theta, s=1)
            plt.ticklabel_format(useOffset=False)
            plt.legend()

            plt.show()
        
        return w0, gamma, A, phi

    def complexLorentzian (self, w, w0, gamma, A, phi, c0, c1, m0, m1):
        L = A * np.exp(phi*1j) / (w-w0 + gamma/2*1j) + c0 + c1*1j + (m0 + m1*1j)*w
        return L
    
    """
    Single lorentzian fitting using least squares
    """
    def single_lorentzian_fitting(self, w0_guess=0, gamma_guess=0, A_guess=0, phi_guess=0, c0_guess=0, c1_guess=0, m0_guess=0, m1_guess=0):
        y_fit = self.real + self.img * 1j
        model = Model(self.complexLorentzian)
        if w0_guess == 0:
            w0_guess = self.freq[0] + (self.freq[-1] - self.freq[0]) / 2
        if gamma_guess == 0:
            gamma_guess = (self.freq[-1] - self.freq[0]) / 4
        # if A_guess == 0:
        #     amp = np.sqrt(self.real ** 2 + self.img ** 2)
        #     A_guess = (np.nanmax(amp) - np.nanmin(amp)) / 2
        # if phi_guess == 0:
        #     phi_guess = np.pi / 2
        
        # print(f"{w0_guess}\n{gamma_guess}\n{A_guess}\n{phi_guess}")

        # model.set_param_hint('A', min = 0)
        # model.set_param_hint('phi', min = 0)
        params = model.make_params(w0 = w0_guess, gamma = gamma_guess, A = A_guess, phi = phi_guess,
                                  c0=c0_guess, c1=c1_guess, m0=m0_guess, m1 = m1_guess)
 
        out = model.fit(y_fit, params=params, w = self.freq)
        # report_fit(out)
        best_vals = out.best_values
        w0, gamma, A, phi, c0, c1, m0, m1 = best_vals['w0'], best_vals['gamma'], best_vals['A'], best_vals['phi'], best_vals['c0'], best_vals['c1'], best_vals['m0'], best_vals['m1']
        if A < 0:
            A = -A
            phi = phi + np.pi
        
        while phi < 0:
            phi = phi + 2 * np.pi
        
        while phi > 2 * np.pi:
            phi = phi - 2 * np.pi

        return w0, gamma, A, phi, c0, c1, m0, m1
    

if __name__ == "__main__":
    def bubbleSort(array):
        # loop to access each array element
        for i in range(len(array)):
            # loop to compare array elements
            for j in range(0, len(array) - i - 1):
            # compare two adjacent elements
            # change > to < to sort in descending order
                if float(array[j].split('_')[1]) >= float(array[j + 1].split('_')[1]):
                    # swapping elements if elements
                    # are not in the intended order
                    temp = array[j]
                    array[j] = array[j+1]
                    array[j+1] = temp
    def openbinfile(filename):
        dat = open(filename, 'r')
        data = np.fromfile(dat, dtype=np.dtype('>f8'))  # the raw data are binary files, so this imports them
        split_index = int((len(data) - 1) / 3)
        frequency = data[1: split_index + 1]
        x = data[split_index + 1: 2 * split_index + 1]
        y = data[2 * split_index + 1: 3 * split_index + 1]
        return frequency * 1000, x, y
    
    dirnames = [
    # r"D:\Alexander\Bi2Se3-2407A\corner\cooled",
    # r"D:\Alexander\Bi2Se3-2407B\corner\cooled",
    r"D:\Alexander\Bi2Se3-2407D\corner\cooled"
    ]

    for dirname in dirnames:
        print(dirname)

        ext = ('.bin')
        filename = []

        # iterating over all files
        for files in os.listdir(dirname):
            if files.endswith(ext):
                filename.append(str(files))
            else:
                continue

        bubbleSort(filename)

        i = 0
        for name in filename:
            path = os.path.join(dirname, name)
            f, x, y = openbinfile(path)
            if i == 0:
                freq, real, img = f, x, y
            else:
                freq = np.concatenate([freq, f])
                real = np.concatenate([real, x])
                img = np.concatenate([img, y])
            i = i + 1
    
    # plt.figure()
    # plt.scatter(freq, np.sqrt(real ** 2 + img ** 2), s=1)

    plt.figure()
    plt.scatter(real, img, s=1)
    
    start = 227500 + 500 - 500
    end = 228500 - 200 + 200
    freq = freq[start : end]
    real = real[start : end]
    img = img[start : end]
    
    plt.axis("scaled")
    plt.scatter(real, img, s=1)
    plt.show()

    start = time.time()

    fit = FitResonances(freq, real, img)
    # w0, gamma, A, phi = fit.arkady()
    # print(f"{w0}\n{gamma}\n{A}\n{phi}\n")
    # fit.plot_lorentzian(w0, gamma, A, phi)
    
    # end = time.time()
    # print(f"{end - start} seconds")
    
    # start = time.time()

    w0, gamma, A, phi, c0, c1, m0, m1 = fit.single_lorentzian_fitting(2409177.8600734463, 283.75864984633785, -0.61220341769469382, -0.79334269391175793)
    print(f"{w0}\n{gamma}\n{A}\n{phi}\n")
    fit.plot_lorentzian(w0, gamma, A, phi, c0, c1, m0, m1)

    end = time.time()
    print(f"{end - start} seconds")