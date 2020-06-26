import pygad
import numpy
import math
import sys
import os
from os import walk
import timeit
import csv
import statistics 


#Function for running the GA
def run_ga(
    callback_generation,
    fitness_function,
    graph, 
    it,
    num_generations, 
    sol_per_pop,
    initial_population,
    parent_selection_type, 
    crossover_type,
    mutation_type,
    mutation_percent_genes,
    mutation_num_genes,
    K_tournament,
    input_filename,
    output_filename,
    mutation_percentage,
    crossover_percentage,
    vertices,
    opt,
    selected_vertices=None,
    #search_percentage=0,
    seed=1256823
    ):

    fitness_function = fitness_function

    num_genes = vertices

    init_range_low = 0
    init_range_high = 1

    random_mutation_min_val = 0
    random_mutation_max_val = 1
    mutation_by_replacement = True
    

    # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
    ga_instance = pygad.GA(num_generations=num_generations,
                           fitness_func=fitness_function,
                           initial_population=initial_population,   
                           sol_per_pop=sol_per_pop,                        
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_by_replacement=mutation_by_replacement,
                           mutation_num_genes=mutation_num_genes,
                           mutation_percent_genes=mutation_percent_genes,
                           callback_generation=callback_generation,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           mutation_percentage = mutation_percentage,
                           crossover_percentage = crossover_percentage,
                           K_tournament=K_tournament,
                           selected_vertices=selected_vertices,
                           #search_percentage=search_percentage,
                           opt=opt,
                           seed=seed)


    
    # Running the GA to optimize the parameters of the function.
    start = timeit.default_timer()
    ga_instance.run()
    stop = timeit.default_timer()
    

    # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
    #ga_instance.plot_result()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    #print("Parameters of the best solution : {solution}".format(solution=solution))
    #print(solution_fitness)
    print("Runtime: " + str((stop-start)*1000))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Number of evaluation used: = {evaluations}".format(evaluations=ga_instance.evaluations))
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.generations_completed))
    #print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    #prediction = numpy.sum(numpy.array(function_inputs)*solution)
    #print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    #if ga_instance.best_solution_generation != -1:
        #print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

    # Saving the GA instance.
    #ga_instance.save(filename=output_filename)
    
    return ga_instance
    
    
    #last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    #print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    #print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    #print("Evaluations = {evaluations}".format(evaluations=ga_instance.evaluations))
    #print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    #last_fitness = ga_instance.cal_pop_fitness()
        

"""
ga_instance = run_ga(
                    fitness_function=fitness_func,
                    graph=graph, 
                    it=it,
                    num_generations = num_generations,
                    sol_per_pop = sol_per_pop,
                    initial_population=None,
                    parent_selection_type = "tournament", 
                    crossover_type = "single_point",
                    mutation_type = "random",
                    mutation_percent_genes = 0.05,
                    mutation_num_genes = None,
                    K_tournament = 2,
                    input_filename=input_filename,
                    output_filename=output_filename,
                    callback_generation=callback_generation,
                    mutation_percentage = 1,
                    crossover_percentage = 1,
                    selected_vertices=select_vertices_to_search(0.1),
                    vertices=vertices,
                    )
"""
"""
#index of file in directory
index = 411
#number of iterations
num_iter = 10    
f = []
dataset = "set0e/"
for (dirpath, dirnames, filenames) in walk("data/" + dataset):
    f.extend(filenames)
    break;
    
def parse_graph(filename):
    graph = []
    data = open(filename,'r').readlines()
    vertices = int(data[0].split(' ')[0])
    for edge in data[1:]:
        edge = edge.strip().split(' ')
        edge = [int(e) for e in edge]
        graph.append((edge[0]-1, edge[1]-1, edge[2]))
    return graph, vertices    
    
num_generations = 1000
sol_per_pop = 64
input_filename = "data/" + dataset + f[index]
graph, vertices = parse_graph(input_filename)
        
def select_vertices_to_search(percentage_selected):
    vertex_weights = []
    for i in range(vertices):
        vertex_weights.append((i, 0, 0))

    for edge in graph:
        vertex_weights[edge[0]] = (edge[0],vertex_weights[edge[0]][0] + edge[2], vertex_weights[edge[0]][0] + 1)
        vertex_weights[edge[1]] = (edge[1],vertex_weights[edge[1]][0] + edge[2], vertex_weights[edge[1]][0] + 1)


    vertex_weights.sort(key=lambda element: (element[1], element[2]))

    num_selected = numpy.uint32(percentage_selected * vertices)
    if num_selected == 0:
        num_selected = 1
        
    selected_vertices = []
    for i in range(num_selected):
        selected_vertices.append(vertex_weights[i][0])
    return selected_vertices


print(vertices)

input_optimum_filename = "data/" + dataset + f[index-1]
data = open(input_optimum_filename,'r').readlines()
opt = int(data[0].strip())
    
def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    fitness = 0
    for edge in graph:
        if solution[edge[0]] != solution[edge[1]]:
            fitness += edge[2]
    return fitness

"""
#function for running the black box comparison
def test_black():
    runtimes = []
    evaluations = []
    generations = []
    index = 401
    for i in range(5):
        index = index + i*2
        num_iter = 10    
        f = []
        dataset = "set0e/"
        for (dirpath, dirnames, filenames) in walk("data/" + dataset):
            f.extend(filenames)
            break;
            
        def parse_graph(filename):
            graph = []
            data = open(filename,'r').readlines()
            vertices = int(data[0].split(' ')[0])
            for edge in data[1:]:
                edge = edge.strip().split(' ')
                edge = [int(e) for e in edge]
                graph.append((edge[0]-1, edge[1]-1, edge[2]))
            return graph, vertices    
            
        num_generations = 1000
        sol_per_pop = 2252
        input_filename = "data/" + dataset + f[index]
        graph, vertices = parse_graph(input_filename)
                
        def select_vertices_to_search(percentage_selected):
            vertex_weights = []
            for i in range(vertices):
                vertex_weights.append((i, 0, 0))

            for edge in graph:
                vertex_weights[edge[0]] = (edge[0],vertex_weights[edge[0]][0] + edge[2], vertex_weights[edge[0]][0] + 1)
                vertex_weights[edge[1]] = (edge[1],vertex_weights[edge[1]][0] + edge[2], vertex_weights[edge[1]][0] + 1)


            vertex_weights.sort(key=lambda element: (element[1], element[2]))

            num_selected = numpy.uint32(percentage_selected * vertices)
            if num_selected == 0:
                num_selected = 1
                
            selected_vertices = []
            for i in range(num_selected):
                selected_vertices.append(vertex_weights[i][0])
            return selected_vertices


        print(vertices)

        input_optimum_filename = "data/" + dataset + f[index-1]
        data = open(input_optimum_filename,'r').readlines()
        opt = int(data[0].strip())
        
        print(opt)
        
        def fitness_func(solution, solution_idx):
            # Calculating the fitness value of each solution in the current population.
            fitness = 0
            for edge in graph:
                if solution[edge[0]] != solution[edge[1]]:
                    fitness += edge[2]
            return fitness

         
        it = 0
        while it < num_iter:
            output_filename = "not used"
            start = timeit.default_timer()
            ga_instance = run_ga(
                    fitness_function=fitness_func,
                    graph=graph, 
                    it=it,
                    num_generations = num_generations,
                    sol_per_pop = sol_per_pop,
                    initial_population=None,
                    parent_selection_type = "tournament", 
                    crossover_type = "uniform",
                    mutation_type = "random",
                    mutation_percent_genes = 0.075,
                    mutation_num_genes = None,
                    K_tournament = 2,
                    input_filename=input_filename,
                    output_filename=output_filename,
                    callback_generation=callback_generation,
                    mutation_percentage = 1,
                    crossover_percentage = 1,
                    vertices=vertices,
                    opt=opt
                    )
            stop = timeit.default_timer()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            if solution_fitness == opt:
                runtimes.append((stop-start)*1000)
                evaluations.append(ga_instance.evaluations)
                generations.append(ga_instance.generations_completed)
                it += 1
         
    
    with open("simple_ga-B-num_eval-" + dataset.split('/')[0] + ".csv", 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([vertices, statistics.mean(evaluations), statistics.stdev(evaluations)])
        
    with open("simple_ga-B-runtime-" + dataset.split('/')[0]  + ".csv", 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([vertices, statistics.mean(runtimes), statistics.stdev(runtimes)])
        
    with open("simple_ga-B-gen-" + dataset.split('/')[0]  + ".csv", 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([vertices, statistics.mean(generations), statistics.stdev(generations)])
         
#test_black()

#function for running the grey box comparison
def test_grey():
    runtimes = []
    evaluations = []
    generations = []
    index = 401
    for i in range(5):
        index = index + i*2
        num_iter = 10    
        f = []
        dataset = "set0e/"
        for (dirpath, dirnames, filenames) in walk("data/" + dataset):
            f.extend(filenames)
            break;
            
        def parse_graph(filename):
            graph = []
            data = open(filename,'r').readlines()
            vertices = int(data[0].split(' ')[0])
            for edge in data[1:]:
                edge = edge.strip().split(' ')
                edge = [int(e) for e in edge]
                graph.append((edge[0]-1, edge[1]-1, edge[2]))
            return graph, vertices    
            
        num_generations = 1000
        sol_per_pop = 2252
        input_filename = "data/" + dataset + f[index]
        graph, vertices = parse_graph(input_filename)
                
        #Select the vertices with most weight and degree        
        def select_vertices_to_search(percentage_selected):
            vertex_weights = []
            for i in range(vertices):
                vertex_weights.append((i, 0, 0))

            for edge in graph:
                vertex_weights[edge[0]] = (edge[0],vertex_weights[edge[0]][0] + edge[2], vertex_weights[edge[0]][0] + 1)
                vertex_weights[edge[1]] = (edge[1],vertex_weights[edge[1]][0] + edge[2], vertex_weights[edge[1]][0] + 1)


            vertex_weights.sort(key=lambda element: (element[1], element[2]))

            num_selected = numpy.uint32(percentage_selected * vertices)
            if num_selected == 0:
                num_selected = 1
                
            selected_vertices = []
            for i in range(num_selected):
                selected_vertices.append(vertex_weights[i][0])
            return selected_vertices


        print(vertices)

        input_optimum_filename = "data/" + dataset + f[index-1]
        data = open(input_optimum_filename,'r').readlines()
        opt = int(data[0].strip())
        
        print(opt)
        
        def fitness_func(solution, solution_idx):
            # Calculating the fitness value of each solution in the current population.
            fitness = 0
            for edge in graph:
                if solution[edge[0]] != solution[edge[1]]:
                    fitness += edge[2]
            return fitness

         
        it = 0
        while it < num_iter:
            output_filename = "not used"
            start = timeit.default_timer()
            ga_instance = run_ga(
                    fitness_function=fitness_func,
                    graph=graph, 
                    it=it,
                    num_generations = num_generations,
                    sol_per_pop = sol_per_pop,
                    initial_population=None,
                    parent_selection_type = "tournament", 
                    crossover_type = "uniform",
                    mutation_type = "random",
                    mutation_percent_genes = 0.05,
                    mutation_num_genes = None,
                    K_tournament = 2,
                    input_filename=input_filename,
                    output_filename=output_filename,
                    callback_generation=callback_generation,
                    mutation_percentage = 1,
                    crossover_percentage = 1,
                    vertices=vertices,
                    selected_vertices=select_vertices_to_search(0.05),
                    opt=opt
                    )
            stop = timeit.default_timer()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            if solution_fitness == opt:
                runtimes.append((stop-start)*1000)
                evaluations.append(ga_instance.evaluations)
                generations.append(ga_instance.generations_completed)
                it += 1
         
    
    with open("simple_ga-W-num_eval-" + dataset.split('/')[0] + ".csv", 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([vertices, statistics.mean(evaluations), statistics.stdev(evaluations)])
        
    with open("simple_ga-W-runtime-" + dataset.split('/')[0]  + ".csv", 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([vertices, statistics.mean(runtimes), statistics.stdev(runtimes)])
        
    with open("simple_ga-W-gen-" + dataset.split('/')[0]  + ".csv", 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([vertices, statistics.mean(generations), statistics.stdev(generations)])
        
test_grey()

#function for running population test
def population_test():
    reliable_found = False
    sol_per_pop = 1
    while not reliable_found:
        sol_per_pop = 2*sol_per_pop
        print("Population size: " + str(sol_per_pop))
        avg_fitness = 0
        
        #if not os.path.exists("results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop)):
            #os.mkdir("results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop))

        for it in range(num_iter):
            output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)

            ga_instance = run_ga(
                                fitness_function=fitness_func,
                                graph=graph, 
                                it=it,
                                num_generations = num_generations,
                                sol_per_pop = sol_per_pop,
                                initial_population=None,
                                parent_selection_type = "tournament", 
                                crossover_type = "uniform",
                                mutation_type = "random",
                                mutation_percent_genes = 0.05,
                                mutation_num_genes = None,
                                K_tournament = 2,
                                input_filename=input_filename,
                                output_filename=output_filename,
                                callback_generation=callback_generation,
                                mutation_percentage = 1,
                                crossover_percentage = 1,
                                selected_vertices=select_vertices_to_search(0.05),
                                vertices=vertices,
                                opt=opt
                                )
                
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            avg_fitness += solution_fitness
            
        avg_fitness = avg_fitness / num_iter
        if avg_fitness >= opt:
            reliable_found = True
    upper_limit = sol_per_pop
    sol_per_pop = sol_per_pop/2
    reliable_found = False

    while not reliable_found and sol_per_pop < upper_limit:
        sol_per_pop = math.ceil(1.1*sol_per_pop)
        if sol_per_pop%2 != 0:
            sol_per_pop = sol_per_pop - 1
        avg_fitness = 0
        print("Population size: " + str(sol_per_pop))

        for it in range(num_iter):
            output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)

            ga_instance = run_ga(
                        fitness_function=fitness_func,
                        graph=graph, 
                        it=it,
                        num_generations = num_generations,
                        sol_per_pop = sol_per_pop,
                        initial_population=None,
                        parent_selection_type = "tournament", 
                        crossover_type = "uniform",
                        mutation_type = "random",
                        mutation_percent_genes = 0.05,
                        mutation_num_genes = None,
                        K_tournament = 2,
                        input_filename=input_filename,
                        output_filename=output_filename,
                        callback_generation=callback_generation,
                        mutation_percentage = 1,
                        crossover_percentage = 1,
                        selected_vertices=select_vertices_to_search(0.05),
                        vertices=vertices,
                        opt=opt
                        )
                
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            avg_fitness += solution_fitness
            
        avg_fitness = avg_fitness / num_iter
        if avg_fitness >= opt:
            reliable_found = True
            print("reliable found with population size: "+ str(sol_per_pop))
             
        sol_per_pop = math.ceil(1.1*sol_per_pop)
    if sol_per_pop > upper_limit:
        print("reliable found with population size: "+ str(upper_limit))

#population_test()


#function for running selection method test
def selection_test(selection_type):
    for it in range(num_iter):
        output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)
        ga_instance = run_ga(
                fitness_function=fitness_func,
                graph=graph, 
                it=it,
                num_generations = num_generations,
                sol_per_pop = sol_per_pop,
                initial_population=None,
                parent_selection_type = selection_type, 
                crossover_type = "uniform",
                mutation_type = "random",
                mutation_percent_genes = 0.05,
                mutation_num_genes = None,
                K_tournament = 2,
                input_filename=input_filename,
                output_filename=output_filename,
                callback_generation=callback_generation,
                mutation_percentage = 1,
                crossover_percentage = 1,
                vertices=vertices,
                opt=opt
                )
#selection_test("rws")

#function for running crossover method
def crossover_test(crossover_type):
    for it in range(num_iter):
        output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)
        ga_instance = run_ga(
                fitness_function=fitness_func,
                graph=graph, 
                it=it,
                num_generations = num_generations,
                sol_per_pop = sol_per_pop,
                initial_population=None,
                parent_selection_type = "tournament", 
                crossover_type = crossover_type,
                mutation_type = "random",
                mutation_percent_genes = 0,
                mutation_num_genes = None,
                K_tournament = 2,
                input_filename=input_filename,
                output_filename=output_filename,
                callback_generation=callback_generation,
                mutation_percentage = 1,
                crossover_percentage = 1,
                vertices=vertices,
                opt=opt
                )
        
#crossover_test("two_points")
    
#function for running mutation test
def mutation_test(mutation_percent_genes):
    for it in range(num_iter):
        output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)
        ga_instance = run_ga(
                fitness_function=fitness_func,
                graph=graph, 
                it=it,
                num_generations = num_generations,
                sol_per_pop = sol_per_pop,
                initial_population=None,
                parent_selection_type = "tournament", 
                crossover_type = "uniform",
                mutation_type = "random",
                mutation_percent_genes = mutation_percent_genes,
                mutation_num_genes = None,
                K_tournament = 2,
                input_filename=input_filename,
                output_filename=output_filename,
                callback_generation=callback_generation,
                mutation_percentage = 1,
                crossover_percentage = 1,
                vertices=vertices,
                opt=opt
                )
        
#population_test()
#mutation_test(0.0)

#function for running the amount k tournament
def tournament_test(k_tournament):
    for it in range(num_iter):
        output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)
        ga_instance = run_ga(
                fitness_function=fitness_func,
                graph=graph, 
                it=it,
                num_generations = num_generations,
                sol_per_pop = sol_per_pop,
                initial_population=None,
                parent_selection_type = "tournament", 
                crossover_type = "uniform",
                mutation_type = "random",
                mutation_percent_genes = 0.05,
                mutation_num_genes = None,
                K_tournament = k_tournament,
                input_filename=input_filename,
                output_filename=output_filename,
                callback_generation=callback_generation,
                mutation_percentage = 1,
                crossover_percentage = 1,
                vertices=vertices,
                )
#tournament_test(2)
  
#function for running mutation percentage test  
def mutation_percentage_test(mutation_percentage):
    for it in range(num_iter):
        output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)
        ga_instance = run_ga(
                fitness_function=fitness_func,
                graph=graph, 
                it=it,
                num_generations = num_generations,
                sol_per_pop = sol_per_pop,
                initial_population=None,
                parent_selection_type = "tournament", 
                crossover_type = "uniform",
                mutation_type = "random",
                mutation_percent_genes = 0.05,
                mutation_num_genes = None,
                K_tournament = 2,
                input_filename=input_filename,
                output_filename=output_filename,
                callback_generation=callback_generation,
                mutation_percentage = mutation_percentage,
                crossover_percentage = 1,
                vertices=vertices,
                opt=opt
                )


#mutation_percentage_test(0.8)

#function for running crossover percentage test
def crossover_percentage_test(crossover_percentage):
    for it in range(num_iter):
        output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)
        ga_instance = run_ga(
                fitness_function=fitness_func,
                graph=graph, 
                it=it,
                num_generations = num_generations,
                sol_per_pop = sol_per_pop,
                initial_population=None,
                parent_selection_type = "tournament", 
                crossover_type = "uniform",
                mutation_type = "random",
                mutation_percent_genes = 0.05,
                mutation_num_genes = None,
                K_tournament = 2,
                input_filename=input_filename,
                output_filename=output_filename,
                callback_generation=callback_generation,
                mutation_percentage = 1,
                crossover_percentage = crossover_percentage,
                vertices=vertices,
                opt=opt
                )
                
#crossover_percentage_test(0.6)
                

#function for 
def select_test(percentage_selected):
    for it in range(num_iter):
        output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)

        ga_instance = run_ga(
                    fitness_function=fitness_func,
                    graph=graph, 
                    it=it,
                    num_generations = num_generations,
                    sol_per_pop = sol_per_pop,
                    initial_population=None,
                    parent_selection_type = "tournament", 
                    crossover_type = "uniform",
                    mutation_type = "random",
                    mutation_percent_genes = 0.05,
                    mutation_num_genes = None,
                    K_tournament = 2,
                    input_filename=input_filename,
                    output_filename=output_filename,
                    callback_generation=callback_generation,
                    mutation_percentage = 1,
                    crossover_percentage = 1,
                    vertices=vertices,
                    selected_vertices=select_vertices_to_search(percentage_selected),
                    opt=opt
                    #search_percentage=search_percentage,
                    )

#select_test(0.2)



"""
for it in range(num_iter):
    output_filename = "results/" + dataset + f[index].split(".", 1)[0] + "_results_" + str(num_generations) + "_" + str(sol_per_pop) + "/" + str(it)
    
    ga_instance = pygad.load(output_filename)
    
    ga_instance.plot_result()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    
    avg_generation += ga_instance.best_solution_generation
   

avg_generation = avg_generation / num_iter

print("Average generations to find best solution: {best_solution_generation}".format(best_solution_generation=avg_generation))
print("Average fitness value of the best solution = {solution_fitness}".format(solution_fitness=avg_fitness))
"""
