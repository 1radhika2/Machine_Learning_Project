from random import sample
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv('GALEX_data-extended-feats.csv')
X = dataset.iloc[:,1:24].values
y = dataset.iloc[:,0].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .3,random_state = 2)

from sklearn.preprocessing import StandardScaler as SC
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test) 
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

model = Sequential()
model.add(Dense(4, input_dim = X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

def genetic_algorithm(model,X_train,y_train,X_test,y_test):
  solution_per_population = 8
  number_parents_mating = 4
  number_of_generations = 120
  mutation_percent = 10
  initial_population_weights = []
  first_layer_biases = model.layers[0].get_weights()[1]
  second_layer_biases = model.layers[2].get_weights()[1]
  for current_solution in np.arange(0,solution_per_population):
    HL1_neurons = 4
    input_HL1_weights = np.random.uniform(low=-0.1, high=0.1,size=(X_train.shape[1], HL1_neurons))
    output_neurons = 3
    HL2_output_weights = np.random.uniform(low=-0.1, high=0.1,size=(HL1_neurons, output_neurons))
    initial_population_weights.append(np.array([input_HL1_weights,HL2_output_weights]))
  population_weights_matrix = np.array(initial_population_weights)
  population_weights_vector = matrix_to_vector(population_weights_matrix)
  accuracy = np.empty(shape=(number_of_generations))
  for generation in range(number_of_generations):
    population_weights_matrix = vector_to_matrix(population_weights_vector,population_weights_matrix)
    fit = fitness(population_weights_matrix,X_train,y_train,activation="sigmoid")
    accuracy[generation] = fit[0]
    parents = select_mating_pool(population_weights_vector,fit.copy(),number_parents_mating)
    offspring_crossover = crossover(parents,offspring_size=(population_weights_vector.shape[0]-parents.shape[0], population_weights_vector.shape[1]))
    offspring_mutation = mutation(offspring_crossover,mutation_percent=mutation_percent)
    population_weights_vector[0:parents.shape[0], :] = parents
    population_weights_vector[parents.shape[0]:, :] = offspring_mutation
  population_weights_matrix = vector_to_matrix(population_weights_vector,population_weights_matrix)
  acc, predictions = predict(population_weights_matrix[0,:],X_test,y_test, activation="sigmoid")
  print("Accuracy of the best solution is : ", acc)
  model.layers[0].set_weights([population_weights_matrix[0,:][0],first_layer_biases])
  model.layers[2].set_weights([population_weights_matrix[0,:][1],second_layer_biases])
  plt.plot(accuracy, linewidth=5, color="black")
  plt.xlabel("Iterations", fontsize=20)
  plt.ylabel("Fitness", fontsize=20)
  plt.xticks(np.arange(0, 51, 10), fontsize=10)
  plt.yticks(np.arange(0, 121, 5), fontsize=10)
  plt.show()
  

def predict(weights_matrix, data_inputs, data_outputs, activation):
    predictions = np.zeros(shape=(data_inputs.shape[0]))
    for sample_index in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_index, :]
        for current_weights in weights_matrix:
            r1 = np.matmul(a=r1, b=current_weights)
            r1 = sigmoid(r1)
        predicted_label = np.where(r1 == np.max(r1))[0][0]
        predictions[sample_index] = predicted_label
    correct_predict = np.where(predictions == data_outputs)[0].size
    acc = (correct_predict/data_outputs.size)*100
    return acc, predictions

def sigmoid(input):
    return 1.0 / (1.0 + np.exp(-1 * input))

def relu(input):
    result = input
    result[input < 0] = 0
    return result

def fitness(weights_matrix, data_inputs, data_outputs, activation="relu"):
    acc = np.empty(shape=(weights_matrix.shape[0]))
    for solution_index in range(weights_matrix.shape[0]):
        acc[solution_index], _ = predict(weights_matrix[solution_index, :], data_inputs, data_outputs, activation=activation)
    return acc
       
def matrix_to_vector(matrix_population_weights):
    population_weights_vector = []
    for solution_index in range(matrix_population_weights.shape[0]):
        current_vector = []
        for layer_index in range(matrix_population_weights.shape[1]):
            current_vector.extend(np.reshape(matrix_population_weights[solution_index, layer_index], newshape=(matrix_population_weights[solution_index, layer_index].size)))
        population_weights_vector.append(current_vector)
    return np.array(population_weights_vector)

def vector_to_matrix(vector_population_weights, matrix_population_weights):
    matrix_weights = []
    for solution_index in range(matrix_population_weights.shape[0]):
        start = 0
        end = 0
        for layer_index in range(matrix_population_weights.shape[1]):
            end += matrix_population_weights[solution_index, layer_index].size
            matrix_weights.append(np.reshape(vector_population_weights[solution_index, start:end], newshape=(matrix_population_weights[solution_index, layer_index].shape)))
            start = end
    return np.reshape(matrix_weights, newshape=matrix_population_weights.shape)

def select_mating_pool(population, fitness, number_of_parents):
  parents = np.empty((number_of_parents, population.shape[1]))
  for parent_num in range(number_of_parents):
    maximum_fitness_index = np.where(fitness == np.max(fitness))[0][0]
    parents[parent_num, :] = population[maximum_fitness_index, :]
    fitness[maximum_fitness_index] = -99999999999
  return parents

def crossover(parents, offspring_size):
  offspring = np.empty(offspring_size)
  crossover_point = np.uint32(offspring_size[1]/2)
  for k in range(offspring_size[0]):
    parent1_index = k%parents.shape[0]
    parent2_index = (k+1)%parents.shape[0]
    offspring[k, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
    offspring[k, crossover_point:] = parents[parent2_index, crossover_point:]
  return offspring

def mutation(offspring_crossover, mutation_percent): 
    mutation_indices = np.array(sample(range(0, offspring_crossover.shape[1]),np.uint32((mutation_percent*offspring_crossover.shape[1])/100)))
    for index in range(offspring_crossover.shape[0]): 
      offspring_crossover[index, mutation_indices] += np.random.uniform(-1.0, 1.0, 1)
    return offspring_crossover


genetic_algorithm(model,X_train,y_train,X_test,y_test)