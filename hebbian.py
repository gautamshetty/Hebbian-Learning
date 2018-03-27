import numpy as np
import Tkinter as Tk
import matplotlib
import scipy.misc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
from numpy import linalg as LA
from scipy import sparse

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys


class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, samples=[[0., 0., 1., 1.], [0., 1., 0., 1.]], targets=[[0., 1., 1., 0.]]):
        # Note: input samples are assumed to be in column order.
        # This means that each column of the samples matrix is representing
        # a sample point
        # The default values for samples and targets represent an exclusive or
        self.samples = np.array(samples)

        if targets != None:
            self.targets = np.array(targets)
        else:
            self.targets = None


nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.1,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 3, "activation_function": "hardlimit"}],  # list of dictionaries
    "data_set": ClDataSet(),
    'number_of_classes': 3,
    'number_of_samples_in_each_class': 3,
    'sample_size': ClDataSet(),
    'x_coords':[],
    'y_coords':[]
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    """

    def __init__(self, settings={}):
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_inputs,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "layers_specification": self.layers_specification,
                    "x_coords": self.x_coords,
                    "y_coords":self.y_coords
                    }
        self.neural_network = ClNeuralNetwork(self, settings)
        # Make sure that the number of neurons in the last layer is equal to number of classes
        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes

    def run_forward_pass(self, display_input=True, display_output=True,
                         display_targets=True, display_target_vectors=True,
                         display_error=True):
        self.neural_network.calculate_output(self.data_set.samples)

        if display_input:
            print "Input : ", self.data_set.samples
        if display_output:
            print 'Output : ', self.neural_network.output
        if display_targets:
            print "Target (class ID) : ", self.target
        if display_target_vectors:
            print "Target Vectors : ", self.desired_target_vectors
        if self.desired_target_vectors.shape == self.neural_network.output.shape:
            self.error = self.desired_target_vectors - self.neural_network.output
            if display_error:
                print 'Error : ', self.error
        else:
            print "Size of the output is not the same as the size of the target.", \
                "Error cannot be calculated."

    def read_one_image_and_convert_to_vector(self, file_name):
        img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
        #img = img.reshape(-1, 1)  # reshape to column vector and return it
        #img = np.column_stack(img)

        """img = img[~np.all(img == 0, axis=1)]
        #img = img[~np.all(img == 0, axis=0)]
        img = np.column_stack(img)
        img = img[~np.all(img == 0, axis=1)]
        img = np.row_stack(img)"""

        img = img.ravel()
        img = np.where(img > 0, 1, img)
        return img

    """
        Read the images and populate the samples data set.
        Initialize the target vectors and weights in the experiment reset method.

    """
    def create_samples(self):

        file_list = []
        sampleSize = []
        fileDirectory = "mnist_images"
        noOfImages = 0
        for k in range(10):
            print k
            noOfImages = 0
            for fileName in os.listdir(fileDirectory):  # "." means current directory
                if fileName.startswith(str(k)+'_'):
                    imgVector = self.read_one_image_and_convert_to_vector(fileDirectory+"/"+fileName)
                    file_list.append(imgVector)
                    noOfImages += 1

            sampleSize.append(noOfImages)

        self.data_set.samples = np.stack(file_list, axis=0)
        self.data_set.sample_size = np.stack(sampleSize)

        self.neural_network.layers[-1].number_of_inputs_to_layer = self.data_set.samples.shape[1]

        self.reset()

        print "samples created"

    """
        Reset the target vectors and weights to initial values.
        Also reset the x & y co-ordinates to empty to reset the graph.
    """
    def reset(self):

        targetVectors = []

        for k in range(10):
            targetZeroVector = np.ones(self.neural_network.layers[-1].number_of_inputs_to_layer, int)
            targetZeroVector[k] = -1
            targetVectors.append(targetZeroVector)

        self.desired_target_vectors = targetVectors

        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes = len(targetVectors)

        self.neural_network.randomize_weights()

        self.x_coords = []
        self.y_coords = []

    def adjust_weights(self, learning_method):

        if (learning_method == "Delta Rule"):
            self.neural_network.adjust_weights_delta_rule(self.data_set.samples, self.data_set.sample_size,
                                                self.desired_target_vectors, learning_method)
        else:
            self.neural_network.adjust_weights(self.data_set.samples, self.data_set.sample_size,
                                               self.desired_target_vectors, learning_method)

class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        #
        self.nn_experiment = nn_experiment
        self.number_of_classes = self.nn_experiment.number_of_classes
        self.xmin = -2
        self.xmax = 2
        self.ymin = -2
        self.ymax = 2
        self.master.update()
        self.number_of_samples_in_each_class = self.nn_experiment.number_of_samples_in_each_class
        self.learning_rate = self.nn_experiment.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.step_size = 0.02
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")
        self.learning_method = "Select"

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)

        """self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = plt.subplot(111)"""

        """
            Show the graph with the initial x & y co-ordinates.
        """
        self.refreshGraph()

        """self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)"""

        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Set up the sliders
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        """
            The various Hebbian learning rules available.
        """
        self.label_for_entry_box = Tk.Label(self.buttons_frame, text="Hebbian rules", justify="center")
        self.label_for_entry_box.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_method_variable = Tk.StringVar()
        self.learning_method_dropdown = Tk.OptionMenu(self.buttons_frame, self.learning_method_variable,
                                                      "Select", "Filtered Learning", "Delta Rule", "Unsupervised Hebb",
                                                      command=lambda event: self.learning_method_dropdown_callback())
        self.learning_method_variable.set("Select")

        self.learning_method_dropdown.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.print_nn_parameters_button = Tk.Button(self.buttons_frame,
                                                    text="Print NN Parameters",
                                                    bg="yellow", fg="red",
                                                    command=lambda: self.print_nn_parameters_button_callback())
        self.print_nn_parameters_button.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.reset_button = Tk.Button(self.buttons_frame,
                                                  text="Reset",
                                                  bg="yellow", fg="red",
                                                  command=lambda: self.reset_button_callback())
        self.reset_button.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.initialize()

    def initialize(self):
        self.nn_experiment.create_samples()
        self.nn_experiment.neural_network.randomize_weights()

    def display_samples_on_image(self):
        # Display the samples for each class
        for class_index in range(0, self.number_of_classes):
            self.axes.scatter(self.nn_experiment.data_set.samples[0, class_index * self.number_of_samples_in_each_class: \
                (class_index + 1) * self.number_of_samples_in_each_class],
                              self.nn_experiment.data_set.samples[1, class_index * self.number_of_samples_in_each_class: \
                                  (class_index + 1) * self.number_of_samples_in_each_class],
                              c=self.sample_points_colors(class_index * (1.0 / self.number_of_classes)),
                              marker=(3 + class_index, 1, 0), s=50)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    def refresh_display(self):
        self.refreshGraph()

    """
        Displays a bar graph according to the x & y co-ordinates.
    """
    def refreshGraph(self):

        print "refresh display"

        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = plt.subplot(111)

        y_co_ord = self.nn_experiment.y_coords
        # std_men = (2, 3, 4, 1, 2)
        fig, ax = plt.subplots()

        x_co_ord = self.nn_experiment.x_coords
        bar_width = 0.10

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        error_graph = self.axes.bar(x_co_ord, y_co_ord, bar_width)

        print "bar graph plotted"
        print error_graph

        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Errors')
        self.axes.set_title('Graph of Errors in Hebb Rules Learning')
        self.axes.legend()

        self.axes.relim()
        plt.draw()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

    def display_neighborhoods(self):
        """self.class_ids = []
        for x, y in np.stack((self.xx.ravel(), self.yy.ravel()), axis=-1):
            output = self.nn_experiment.neural_network.calculate_output(np.array([x, y]))
            self.class_ids.append(output.dot(self.convert_binary_to_integer))
        self.class_ids = np.array(self.class_ids)
        self.class_ids = self.class_ids.reshape(self.xx.shape)
        self.axes.cla()
        self.axes.pcolormesh(self.xx, self.yy, self.class_ids, cmap=self.neighborhood_colors)
        self.display_output_nodes_net_boundaries()
        self.display_samples_on_image()"""
        """mu, sigma = 100, 15
        x = mu + sigma * np.random.randn(10000)

        # the histogram of the data
        n, bins, patches = self.plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

        plt.xlabel('Epoch')
        plt.ylabel('Errors')
        plt.title('Histogram of Hebb Rule Learning')
        plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        plt.show()"""
        #self.displayGraph()

    def display_output_nodes_net_boundaries(self):
        output_layer = self.nn_experiment.neural_network.layers[-1]
        for output_neuron_index in range(output_layer.number_of_neurons):
            w1 = output_layer.weights[output_neuron_index][0]
            w2 = output_layer.weights[output_neuron_index][1]
            w3 = output_layer.weights[output_neuron_index][2]
            if w1 == 0 and w2 == 0:
                data = [(0, 0), (0, 0), 'r']
            elif w1 == 0:
                data = [(self.xmin, self.xmax), (float(w3) / w2, float(w3) / w2), 'r']
            elif w2 == 0:
                data = [(float(-w3) / w1, float(-w3) / w1), (self.ymin, self.ymax), 'r']
            else:
                data = [(self.xmin, self.xmax),  # in form of (x1, x2), (y1, y2)
                        ((-w3 - float(w1 * self.xmin)) / w2,
                         (-w3 - float(w1 * self.xmax)) / w2), 'r']
            self.axes.plot(*data)

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.refresh_display()

    def number_of_classes_slider_callback(self):
        self.number_of_classes = self.number_of_classes_slider.get()
        self.nn_experiment.number_of_classes = self.number_of_classes
        self.nn_experiment.neural_network.layers[-1].number_of_neurons = self.number_of_classes
        self.nn_experiment.neural_network.randomize_weights()
        self.initialize()
        self.refresh_display()

    def number_of_samples_slider_callback(self):
        self.number_of_samples_in_each_class = self.number_of_samples_slider.get()
        self.nn_experiment.number_of_samples_in_each_class = self.number_of_samples_slider.get()
        self.nn_experiment.create_samples()
        self.refresh_display()

    def create_new_samples_bottun_callback(self):
        temp_text = self.create_new_samples_bottun.config('text')[-1]
        self.create_new_samples_bottun.config(text='Please Wait')
        self.create_new_samples_bottun.update_idletasks()
        self.nn_experiment.create_samples()
        self.refresh_display()
        self.create_new_samples_bottun.config(text=temp_text)
        self.create_new_samples_bottun.update_idletasks()

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        #for k in range(10):
        self.nn_experiment.adjust_weights(self.learning_method)
        self.refresh_display()
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()

    def learning_method_dropdown_callback(self):
        self.learning_method = self.learning_method_variable.get()

    def randomize_weights_button_callback(self):
        temp_text = self.randomize_weights_button.config('text')[-1]
        self.randomize_weights_button.config(text='Please Wait')
        self.randomize_weights_button.update_idletasks()
        self.nn_experiment.neural_network.randomize_weights()
        # self.nn_experiment.neural_network.display_network_parameters()
        # self.nn_experiment.run_forward_pass()
        self.refresh_display()
        self.randomize_weights_button.config(text=temp_text)
        self.randomize_weights_button.update_idletasks()

    def print_nn_parameters_button_callback(self):
        temp_text = self.print_nn_parameters_button.config('text')[-1]
        self.print_nn_parameters_button.config(text='Please Wait')
        self.print_nn_parameters_button.update_idletasks()
        self.nn_experiment.neural_network.display_network_parameters()
        self.refresh_display()
        self.print_nn_parameters_button.config(text=temp_text)
        self.print_nn_parameters_button.update_idletasks()

    def reset_button_callback(self):

        self.nn_experiment.reset()
        self.refresh_display()

neural_network_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.1,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 3,
                              "activation_function": "hardlimit"}]  # list of dictionaries
}

class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    """

    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        self.input_index = 0
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self, min=-0.1, max=0.1):
        # randomize weights for all the connections in the network
        for layer in self.layers:
            layer.randomize_weights(self.min_initial_weights, self.max_initial_weights)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)
        self.output = output
        return self.output

    """
        Adjust weights according to the Filtered Learning or Unsupervised learning as selected.
    """
    def adjust_weights(self, input_samples, sample_size, desired_target_vectors, learning_method):

        """
        Decay term
        """
        gamma_term = 0.001

        for layer_index, layer in enumerate(self.layers):

            """
                For each of digits, add the multiplication of target, input and learning rate to the weights.
            """

            sample_index = 0
            target_vector = desired_target_vectors[self.input_index]
            weight = layer.weights[self.input_index]

            for sample_index in range(sample_size[self.input_index]):

                input_row = input_samples[sample_index]

                input_row = np.column_stack(input_row)

                target_input = np.multiply(self.learning_rate, input_row)
                target_input = target_input.dot(target_vector)

                if (learning_method == "Filtered Learning"):
                    weight = np.multiply(gamma_term, weight)

                weight = np.add(weight, target_input)

                sample_index += 1

            """
                After adding all the inputs, weights and learning rate product.
                Multiply the input with the sum to check if it matches or there is error.
            """
            weight = np.multiply(weight, input_row)


            """
                Since the inputs are not orthogonal.
                Calculate the error to be shown on graph.
            """
            difference = np.subtract(target_vector, weight)
            difference = np.multiply(100, difference)
            error_rate = difference / target_vector

            error_mean = np.mean(error_rate)

            """
                Set the x & y co-ordinates to be shown on the bar graph.
            """
            self.experiment.x_coords.append(self.input_index)
            self.experiment.y_coords.append(error_mean/100)

        """
            When the number of digits or neurons are reached, start from first.
        """
        self.input_index = (self.input_index + 1) % self.experiment.number_of_classes

        print "X Co-ords", self.experiment.x_coords
        print "Y Co-ords", self.experiment.y_coords

    """
        Adjust weights according to the Delta Rule learning rule.
    """
    def adjust_weights_delta_rule(self, input_samples, sample_size, desired_target_vectors, learning_method):

        for layer_index, layer in enumerate(self.layers):

            sample_index = 0
            target_vector = desired_target_vectors[self.input_index]
            weight = layer.weights[self.input_index]

            target_term = 0
            for sample_index in range(sample_size[self.input_index]):

                """
                    Calculate the actual output according to the pure linear activation function.
                """
                input_row = input_samples[sample_index]

                input_row = np.column_stack(input_row)

                actual_output = np.dot(weight, input_row)

                actual_output = np.where(actual_output < 0, 0, actual_output)

                """
                    Take difference of target and actual output, and multiply the result by input and learning rate.
                """
                actual_output = np.row_stack(actual_output)

                target_actual = np.subtract(target_vector, actual_output)

                input_row = np.column_stack(input_row)

                target_actual = np.dot(target_actual, input_row)
                target_actual = np.multiply(self.learning_rate, target_actual)

                target_term += target_actual

                sample_index += 1

            """
                Add the product of the output difference, input and learning rate to the weight.
            """
            weight += target_term

            """
                Since the inputs are not orthogonal.
                Calculate the error to be shown on graph.
            """
            difference = np.subtract(target_vector, weight)
            difference = np.multiply(100, difference)
            error_rate = difference / target_vector

            """
                Set the x & y co-ordinates to be shown on the bar graph.
            """
            error_mean = np.mean(error_rate)
            self.experiment.x_coords.append(self.input_index)
            self.experiment.y_coords.append(error_mean / 100)


single_layer_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs_to_layer": 2,  # number of input signals
    "number_of_neurons": 2,  # number of neurons in the layer
    "activation_function": "hardlimit"  # default activation function
}


class ClSingleLayer:
    """
    This class presents a single layer of neurons
    """

    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        self.randomize_weights()

    def randomize_weights(self, min_initial_weights=None, max_initial_weights=None):
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        self.weights = np.zeros(self.number_of_neurons, int)#np.random.uniform(min_initial_weights, max_initial_weights, (self.number_of_neurons, self.number_of_inputs_to_layer + 1))

    def calculate_output(self, input_values):
        # Calculate the output of the layer, given the input signals
        # NOTE: Input is assumed to be a column vector. If the input
        # is given as a matrix, then each column of the input matrix is assumed to be a sample
        if len(input_values.shape) == 1:
            net = self.weights.dot(np.append(input_values, 1))
        else:
            net = self.weights.dot(np.vstack([input_values, np.ones((1, input_values.shape[1]), float)]))
        if self.activation_function == 'linear':
            self.output = net
        if self.activation_function == 'sigmoid':
            self.output = sigmoid(net)
        if self.activation_function == 'hardlimit':
            np.putmask(net, net > 0, 1)
            np.putmask(net, net <= 0, 0)
            self.output = net
        return self.output


if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights": -0.1,  # minimum initial weight
        "max_initial_weights": 0.1,  # maximum initial weight
        "number_of_inputs": 2,  # number of inputs to the network
        "learning_rate": 0.1,  # learning rate
        "layers_specification": [{"number_of_neurons": 3, "activation_function": "hardlimit"}],  # list of dictionaries
        "data_set": ClDataSet(),
        'number_of_classes': 2,
        'number_of_samples_in_each_class': 3
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("Hebbian Learning")
    main_frame.geometry('640x480')
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()