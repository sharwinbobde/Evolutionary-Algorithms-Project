import numpy
import random
import matplotlib.pyplot
import pickle

class GA:
    def __init__(self, 
                 num_generations, 
                 fitness_func,
                 crossover_percentage,
                 mutation_percentage,
                 initial_population=None,
                 sol_per_pop=None, 
                 num_genes=None,
                 init_range_low=-4,
                 init_range_high=4,
                 parent_selection_type="sss",
                 K_tournament=3,
                 crossover_type="single_point",
                 mutation_type="random",
                 mutation_by_replacement=False,
                 mutation_percent_genes=10,
                 mutation_num_genes=None,
                 random_mutation_min_val=-1.0,
                 random_mutation_max_val=1.0,
                 callback_generation=None,
                 selected_vertices=None,
                 #search_percentage=0.0,
                 seed=1256823,
                 opt=-1):

        """
        The constructor of the GA class accepts all parameters required to create an instance of the GA class. It validates such parameters.

        num_generations: Number of generations.
        num_parents_mating: Number of solutions to be selected as parents in the mating pool.

        fitness_func: Accepts a function that must accept 2 parameters (a single solution and its index in the population) and return the fitness value of the solution. Available starting from PyGAD 1.0.17 until 1.0.20 with a single parameter representing the solution. Changed in PyGAD 2.0.0 and higher to include the second parameter representing the solution index.

        initial_population: A user-defined initial population. It is useful when the user wants to start the generations with a custom initial population. It defaults to None which means no initial population is specified by the user. In this case, PyGAD creates an initial population using the 'sol_per_pop' and 'num_genes' parameters. An exception is raised if the 'initial_population' is None while any of the 2 parameters ('sol_per_pop' or 'num_genes') is also None.
        sol_per_pop: Number of solutions in the population. 
        num_genes: Number of parameters in the function.

        init_range_low: The lower value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20 and higher.
        init_range_high: The upper value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20.
        # It is OK to set the value of any of the 2 parameters ('init_range_high' and 'init_range_high') to be equal, higher or lower than the other parameter (i.e. init_range_low is not needed to be lower than init_range_high).

        parent_selection_type: Type of parent selection.
        keep_parents: If 0, this means the parents of the current population will not be used at all in the next population. If -1, this means all parents in the current population will be used in the next population. If set to a value > 0, then the specified value refers to the number of parents in the current population to be used in the next population. In some cases, the parents are of high quality and thus we do not want to loose such some high quality solutions. If some parent selection operators like roulette wheel selection (RWS), the parents may not be of high quality and thus keeping the parents might degarde the quality of the population.
        K_tournament: When the value of 'parent_selection_type' is 'tournament', the 'K_tournament' parameter specifies the number of solutions from which a parent is selected randomly.

        crossover_type: Type of the crossover opreator. If  crossover_type=None, then the crossover step is bypassed which means no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
        mutation_type: Type of the mutation opreator. If mutation_type=None, then the mutation step is bypassed which means no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.

        mutation_by_replacement: An optional bool parameter. It works only when the selected type of mutation is random (mutation_type="random"). In this case, setting mutation_by_replacement=True means replace the gene by the randomly generated value. If False, then it has no effect and random mutation works by adding the random value to the gene.

        mutation_percent_genes: Percentage of genes to mutate which defaults to 10%. This parameter has no action if the parameter mutation_num_genes exists.
        mutation_num_genes: Number of genes to mutate which defaults to None. If the parameter mutation_num_genes exists, then no need for the parameter mutation_percent_genes.
        random_mutation_min_val: The minimum value of the range from which a random value is selected to be added to the selected gene(s) to mutate. It defaults to -1.0.
        random_mutation_max_val: The maximum value of the range from which a random value is selected to be added to the selected gene(s) to mutate. It defaults to 1.0.

        callback_generation: If not None, then it accepts a function to be called after each generation. This function must accept a single parameter representing the instance of the genetic algorithm.
        
        selected_vertices: If not None, then the search addition is activated. The selected vertices which will be searched for more optimal solution
        seed: seed for random generator
        opt: optimal fitness to the problem
        """
        
        random.seed(seed)

        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        
        self.crossover_percentage = crossover_percentage
        self.mutation_percentage = mutation_percentage
        
        self.evaluations = 0
        self.fitness = numpy.empty(0)
        self.best_solutions = []
        self.best_idx = []
        
        self.opt = opt
        
        self.selected_vertices = selected_vertices
        #self.search_percentage = search_percentage
        
        # Check if the fitness_func is a function.
        if callable(fitness_func):
            # Check if the fitness function accepts 2 paramaters.
            if (fitness_func.__code__.co_argcount == 2):
                self.fitness_func = fitness_func
            else:
                self.valid_parameters = False
                raise ValueError("The fitness function must accept 2 parameters representing the solution to which the fitness value is calculated and the solution index within the population.\nThe passed fitness function named '{funcname}' accepts {argcount} argument(s).".format(funcname=fitness_func.__code__.co_name, argcount=fitness_func.__code__.co_argcount))
        else:
            self.valid_parameters = False
            raise ValueError("The value assigned to the 'fitness_func' parameter is expected to be of type function by {fitness_func_type} found.".format(fitness_func_type=type(fitness_func)))


        if initial_population is None:
            if (sol_per_pop is None) or (num_genes is None):
                raise ValueError("Error creating the initail population\n\nWhen the parameter initial_population is None, then neither of the 2 parameters sol_per_pop and num_genes can be None at the same time.\nThere are 2 options to prepare the initial population:\n1) Create an initial population and assign it to the initial_population parameter. In this case, the values of the 2 parameters sol_per_pop and num_genes will be deduced.\n2) Allow the genetic algorithm to create the initial population automatically by passing valid integer values to the sol_per_pop and num_genes parameters.")
            elif (type(sol_per_pop) is int) and (type(num_genes) is int):
                # Validating the number of solutions in the population (sol_per_pop)
                if sol_per_pop <= 0:
                    self.valid_parameters = False
                    raise ValueError("The number of solutions in the population (sol_per_pop) must be > 0 but {sol_per_pop} found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n".format(sol_per_pop=sol_per_pop))
                # Validating the number of gene.
                if (num_genes <= 0):
                    self.valid_parameters = False
                    raise ValueError("Number of genes cannot be <= 0 but {num_genes} found.\n".format(num_genes=num_genes))
                # When initial_population=None and the 2 parameters sol_per_pop and num_genes have valid integer values, then the initial population is created.
                # Inside the initialize_population() method, the initial_population attribute is assigned to keep the initial population accessible.
                self.num_genes = num_genes # Number of genes in the solution.
                self.sol_per_pop = sol_per_pop # Number of solutions in the population.
                self.initialize_population(self.init_range_low, self.init_range_high)
            else:
                raise TypeError("The expected type of both the sol_per_pop and num_genes parameters is int but {sol_per_pop_type} and {num_genes_type} found.".format(sol_per_pop_type=type(sol_per_pop), num_genes_type=type(num_genes)))
        elif numpy.array(initial_population).ndim != 2:
            raise ValueError("A 2D list is expected to the initail_population parameter but a {initial_population_ndim}-D list found.".format(initial_population_ndim=numpy.array(initial_population).ndim))
        else:
            self.initial_population = numpy.array(initial_population)
            self.population = self.initial_population # A NumPy array holding the initial population.
            self.num_genes = self.initial_population.shape[1] # Number of genes in the solution.
            self.sol_per_pop = self.initial_population.shape[0]  # Number of solutions in the population.
            self.pop_size = (self.sol_per_pop,self.num_genes) # The population size.
            
        # crossover: Refers to the method that applies the crossover operator based on the selected type of crossover in the crossover_type property.
        # Validating the crossover type: crossover_type
        if (crossover_type == "single_point"):
            self.crossover = self.single_point_crossover
        elif (crossover_type == "two_points"):
            self.crossover = self.two_points_crossover
        elif (crossover_type == "uniform"):
            self.crossover = self.uniform_crossover
        elif (crossover_type is None):
            self.crossover = None
        else:
            self.valid_parameters = False
            raise ValueError("Undefined crossover type. \nThe assigned value to the crossover_type ({crossover_type}) argument does not refer to one of the supported crossover types which are: \n-single_point (for single point crossover)\n-two_points (for two points crossover)\n-uniform (for uniform crossover).\n".format(crossover_type=crossover_type))

        self.crossover_type = crossover_type

        # mutation: Refers to the method that applies the mutation operator based on the selected type of mutation in the mutation_type property.
        # Validating the mutation type: mutation_type
        if (mutation_type == "random"):
            self.mutation = self.random_mutation
        elif (mutation_type == "swap"):
            self.mutation = self.swap_mutation
        elif (mutation_type == "scramble"):
            self.mutation = self.scramble_mutation
        elif (mutation_type == "inversion"):
            self.mutation = self.inversion_mutation
        elif (mutation_type is None):
            self.mutation = None
        else:
            self.valid_parameters = False
            raise ValueError("Undefined mutation type. \nThe assigned value to the mutation_type argument ({mutation_type}) does not refer to one of the supported mutation types which are: \n-random (for random mutation)\n-swap (for swap mutation)\n-inversion (for inversion mutation)\n-scramble (for scramble mutation).\n".format(mutation_type=mutation_type))

        self.mutation_type = mutation_type

        if not (self.mutation_type is None):
            if (mutation_num_genes == None):
                if (mutation_percent_genes < 0 or mutation_percent_genes > 100):
                    self.valid_parameters = False
                    raise ValueError("The percentage of selected genes for mutation (mutation_percent_genes) must be >= 0 and <= 100 inclusive but {mutation_percent_genes=mutation_percent_genes} found.\n".format(mutation_percent_genes=mutation_percent_genes))
            elif (mutation_num_genes <= 0 ):
                self.valid_parameters = False
                raise ValueError("The number of selected genes for mutation (mutation_num_genes) cannot be <= 0 but {mutation_num_genes} found.\n".format(mutation_num_genes=mutation_num_genes))
            elif (mutation_num_genes > self.num_genes):
                self.valid_parameters = False
                raise ValueError("The number of selected genes for mutation (mutation_num_genes) ({mutation_num_genes}) cannot be greater than the number of genes ({num_genes}).\n".format(mutation_num_genes=mutation_num_genes, num_genes=self.num_genes))
            elif (type(mutation_num_genes) is not int):
                self.valid_parameters = False
                raise ValueError("The number of selected genes for mutation (mutation_num_genes) must be a positive integer >= 1 but {mutation_num_genes} found.\n".format(mutation_num_genes=mutation_num_genes))
        else:
            pass

        if not (type(mutation_by_replacement) is bool):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'mutation_by_replacement' parameter is bool but {mutation_by_replacement_type} found.".format(mutation_by_replacement_type=type(mutation_by_replacement)))

        self.mutation_by_replacement = mutation_by_replacement
        
        if self.mutation_type != "random" and self.mutation_by_replacement:
            print("Warning: The mutation_by_replacement parameter is set to True while the mutation_type parameter is not set to random but {mut_type}. Note that the mutation_by_replacement parameter has an effect only when mutation_type='random'.".format(mut_type=mutation_type))

        if (self.mutation_type is None) and (self.crossover_type is None):
            print("Warning: the 2 parameters mutation_type and crossover_type are None. This disables any type of evolution the genetic algorithm can make. As a result, the genetic algorithm cannot find a better solution that the best solution in the initial population.")

        # select_parents: Refers to a method that selects the parents based on the parent selection type specified in the parent_selection_type attribute.
        # Validating the selected type of parent selection: parent_selection_type
        if (parent_selection_type == "sss"):
            self.select_parents = self.steady_state_selection
        elif (parent_selection_type == "rws"):
            self.select_parents = self.roulette_wheel_selection
        elif (parent_selection_type == "sus"):
            self.select_parents = self.stochastic_universal_selection
        elif (parent_selection_type == "random"):
            self.select_parents = self.random_selection
        elif (parent_selection_type == "tournament"):
            self.select_parents = self.tournament_selection
        elif (parent_selection_type == "rank"):
            self.select_parents = self.rank_selection
        else:
            self.valid_parameters = False
            raise ValueError("Undefined parent selection type: {parent_selection_type}. \nThe assigned value to the parent_selection_type argument does not refer to one of the supported parent selection techniques which are: \n-sss (for steady state selection)\n-rws (for roulette wheel selection)\n-sus (for stochastic universal selection)\n-rank (for rank selection)\n-random (for random selection)\n-tournament (for tournament selection).\n".format(parent_selection_type))

        if(parent_selection_type == "tournament"):
            if (K_tournament > self.sol_per_pop):
                K_tournament = self.sol_per_pop
                print("Warining: K of the tournament selection ({K_tournament}) should not be greater than the number of solutions within the population ({sol_per_pop}).\nK will be clipped to be equal to the number of solutions in the population (sol_per_pop).\n".format(K_tournament=K_tournament, sol_per_pop=self.sol_per_pop))
            elif (K_tournament <= 0):
                self.valid_parameters = False
                raise ValueError("K of the tournament selection cannot be <=0 but {K_tournament} found.\n".format(K_tournament=K_tournament))

        self.K_tournament = K_tournament

        # Check if the callback_generation exists.
        if not (callback_generation is None):
            # Check if the callback_generation is a function.
            if callable(callback_generation):
                # Check if the callback_generation function accepts only a single paramater.
                if (callback_generation.__code__.co_argcount == 1):
                    self.callback_generation = callback_generation
                else:
                    self.valid_parameters = False
                    raise ValueError("The callback_generation function must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed callback_generation function named '{funcname}' accepts {argcount} argument(s).".format(funcname=callback_generation.__code__.co_name, argcount=callback_generation.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the 'callback_generation' parameter is expected to be of type function by {callback_generation_type} found.".format(callback_generation_type=type(callback_generation)))
        else:
            self.callback_generation = None

        # The number of completed generations.
        self.generations_completed = 0

        # At this point, all necessary parameters validation is done successfully and we are sure that the parameters are valid.
        self.valid_parameters = True # Set to True when all the parameters passed in the GA class constructor are valid.

        # Parameters of the genetic algorithm.
        self.num_generations = abs(num_generations)
        self.parent_selection_type = parent_selection_type

        # Parameters of the mutation operation.
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_num_genes = mutation_num_genes
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val

        # Even such this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
        self.best_solutions_fitness = [] # A list holding the fitness value of the best solution for each generation.

        self.best_solution_generation = -1 # The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.

    def initialize_population(self, low, high):

        """
        Creates an initial population randomly as a NumPy array. The array is saved in the instance attribute named 'population'.

        low: The lower value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20 and higher.
        high: The upper value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20.
        
        This method assigns the values of the following 3 instance attributes:
            1. pop_size: Size of the population.
            2. population: Initially, holds the initial population and later updated after each generation.
            3. init_population: Keeping the initial population.
        """

        # Population size = (number of chromosomes, number of genes per chromosome)
        self.pop_size = (self.sol_per_pop,self.num_genes) # The population will have sol_per_pop chromosome where each chromosome has num_genes genes.
        # Creating the initial population randomly.
        self.population = numpy.random.randint(low=low, 
                                               high=high+1, 
                                               size=self.pop_size) # A NumPy array holding the initial population.
        self.fitness = self.cal_pop_fitness(self.population)
        
        idx = numpy.argmax(self.fitness)
        if self.selected_vertices != None:
            idx = numpy.argmax(self.fitness)
            sol, fit = self.search(self.population[idx], self.selected_vertices)
            self.fitness[idx] = fit
            self.population[idx] = sol
            """
            num_search_indexes = numpy.uint32(self.search_percentage*self.population.shape[0])
            search_indexes = numpy.array(random.sample(range(0, self.population.shape[0]), num_search_indexes))
            for i in search_indexes
                sol, fit = self.search(self.population[i], self.selected_vertices)
                self.fitness[i] = fit
                self.population[i] = sol
            """
        

        
        # Keeping the initial population in the initial_population attribute.
        self.initial_population = self.population.copy()
    
    #inverts a bit in a solution
    def invert(self, bit):
        return (bit+1) % 2
        
    #function for local search of selected_vertices    
    def search(self, sol, selected):
        if len(selected) == 1:
            fit1 = self.fitness_func(sol, 0)
            sel = selected[0]
            pot_sol = sol.copy()
            pot_sol[sel] = self.invert(pot_sol[sel])
            fit2 = self.fitness_func(pot_sol, 0)
            self.evaluations += 2
            #print(sol)
            #print(pot_sol)
            if fit1 >= fit2:
                return (sol, fit1)
            else:
                return (pot_sol, fit2)
        else:
            new_selected = selected.copy()
            new_selected.pop(0)
            (alt1, fit1) = self.search(sol, new_selected)
            sel = selected[0]
            pot_sol = sol.copy()
            pot_sol[sel] = self.invert(pot_sol[sel])
            new_selected2 = selected.copy()
            new_selected2.pop(0)
            (alt2, fit2) = self.search(pot_sol, new_selected2)
            if fit1 >= fit2:
                return (alt1, fit1)
            else:
                return (alt2, fit2)
    

    def cal_pop_fitness(self, pop):

        """
        Calculating the fitness values of all solutions in the current population. 
        It returns:
            -fitness: An array of the calculated fitness values.
        """
        
        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol_idx, sol in enumerate(pop):
            self.evaluations += 1
            fitness = self.fitness_func(sol, sol_idx)
            pop_fitness.append(fitness)

        pop_fitness = numpy.array(pop_fitness)

        return pop_fitness

    def run(self):

        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """

        generation_same = 0
        last_best = numpy.max(self.fitness)
        if self.valid_parameters == False:
            raise ValueError("ERROR calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

        for generation in range(self.num_generations):
            
            # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
            # Generating offspring using crossover.
            offspring_crossover = self.crossover(self.population,
                                                 offspring_size=(self.sol_per_pop, self.num_genes))

            # Adding some variations to the offspring using mutation.
            offspring_mutation = self.mutation(offspring_crossover)
             
            # Measuring the fitness of each chromosome in the offspring population. 
            fitness_offspring = self.cal_pop_fitness(offspring_mutation)
            
            #concatenating the parents and offspring
            self.population = numpy.concatenate((self.population,offspring_mutation))
            self.fitness = numpy.concatenate((self.fitness, fitness_offspring))
            
            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            idx = numpy.argmax(self.fitness) 
            best_sol = self.population[idx]
            best_fitness = numpy.max(self.fitness)
                
            # Selecting the best parents in the population.
            self.population = self.select_parents(self.fitness, num_parents=self.sol_per_pop*2)
            
            #Save over the worst with the best solution
            self.population[numpy.argmin(self.fitness)] = best_sol 
            self.fitness[numpy.argmin(self.fitness)] = best_fitness

            self.best_solutions_fitness.append(best_fitness)
            idx = numpy.argmax(self.fitness) 
            
            #if the best solution has changed
            if last_best == best_fitness:
                generation_same += 1
            else:
                last_best = best_fitness
                
                if self.selected_vertices != None:
                    idx = numpy.argmax(self.fitness)
                    sol, fit = self.search(self.population[idx], self.selected_vertices)
                    self.fitness[idx] = fit
                    self.population[idx] = sol
                    """num_search_indexes = numpy.uint32(self.search_percentage*self.population.shape[0])
                    search_indexes = numpy.array(random.sample(range(0, self.population.shape[0]), num_search_indexes))
                    for i in search_indexes:
                    sol, fit = self.search(self.population[i], self.selected_vertices)
                    self.fitness[i] = fit
                    self.population[i] = sol
                    """
                #print(best_fitness)
                generation_same = 0
            #print(generation)    
            
            self.best_idx.append(idx)
            best_sol = self.population[idx]
            self.best_solutions.append(best_sol)  
            
            self.generations_completed = generation + 1 # The generations_completed attribute holds the number of the last completed generation.

            # If the callback_generation attribute is not None, then cal the callback function after the generation.
            if not (self.callback_generation is None):
                self.callback_generation(self)
                
            #if a optimal solution has been found    
            if numpy.max(self.fitness) == self.opt or generation_same == 40:
                print("Interrupted")
                self.best_solution_generation = numpy.where(numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][0]
                self.run_completed = True
                break
                
            #print(generation)

        self.best_solution_generation = numpy.where(numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True # Set to True only after the run() method completes gracefully.

    def steady_state_selection(self, fitness, num_parents):

        """
        Selects the parents using the steady-state selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        new_fitness = numpy.empty(num_parents)
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :]
            new_fitness[parent_num] = fitness[fitness_sorted[parent_num]]
        self.fitness = new_fitness
        return parents

    def rank_selection(self, fitness, num_parents):

        """
        Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        new_fitness = numpy.empty(num_parents)
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :]
            new_fitness[parent_num] = fitness[fitness_sorted[parent_num]]
        self.fitness = new_fitness
        return parents

    def random_selection(self, fitness, num_parents):

        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        parents = numpy.empty((num_parents, self.population.shape[1]))

        rand_indices = numpy.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)
        new_fitness = numpy.empty(num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :]
            new_fitness[parent_num] = fitness[rand_indices[parent_num]]
        self.fitness = new_fitness
        return parents

    def tournament_selection(self, fitness, num_parents):

        """
        Selects the parents using the tournament selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        parents = numpy.empty((num_parents, self.population.shape[1]))
        new_fitness = numpy.empty(num_parents)
        for parent_num in range(num_parents):
            rand_indices = numpy.random.randint(low=0.0, high=len(fitness), size=self.K_tournament)
            K_fitnesses = fitness[rand_indices]
            selected_parent_idx = numpy.where(K_fitnesses == numpy.max(K_fitnesses))[0][0]
            parents[parent_num, :] = self.population[rand_indices[selected_parent_idx], :]
            new_fitness[parent_num] = fitness[rand_indices[selected_parent_idx]]
        self.fitness = new_fitness
        return parents

    def roulette_wheel_selection(self, fitness, num_parents):

        """
        Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = numpy.sum(fitness)
        probs = fitness / fitness_sum
        probs_start = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the start values of the ranges of probabilities.
        probs_end = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0
        
        

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = numpy.where(probs == numpy.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        new_fitness = numpy.empty(num_parents);
        
        for parent_num in range(num_parents):
            rand_prob = numpy.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :]
                    new_fitness[parent_num] = fitness[idx]
                    break
                    
        self.fitness = new_fitness
        return parents

    def stochastic_universal_selection(self, fitness, num_parents):

        """
        Selects the parents using the stochastic universal selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = numpy.sum(fitness)
        probs = fitness / fitness_sum
        probs_start = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the start values of the ranges of probabilities.
        probs_end = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the end values of the ranges of probabilities.
        

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = numpy.where(probs == numpy.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        pointers_distance = 1.0 / self.num_parents_mating # Distance between different pointers.
        first_pointer = numpy.random.uniform(low=0.0, high=pointers_distance, size=1) # Location of the first pointer.

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        new_fitness = numpy.empty(num_parents)
        for parent_num in range(num_parents):
            rand_pointer = first_pointer + parent_num*pointers_distance
            for idx in range(probs.shape[0]):
                if (rand_pointer >= probs_start[idx] and rand_pointer < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :]
                    new_fitness[parent_num] = fitness[idx]
                    break
                    
        self.fitness = new_fitness
        return parents

    def single_point_crossover(self, parents, offspring_size):

        """
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        offspring = numpy.copy(parents)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = numpy.random.randint(low=0, high=parents.shape[1], size=1)[0]
        
        
        crossover_offspring = numpy.uint32(self.crossover_percentage*offspring.shape[0])
        offspring_indices = []
        if self.crossover_percentage != 1:
            offspring_indices = numpy.array(random.sample(range(0, offspring.shape[0]), crossover_offspring))
        else:
            offspring_indices = range(offspring.shape[0])

        for k in offspring_indices:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
            
        
        return offspring

    def two_points_crossover(self, parents, offspring_size):

        """
        Applies the 2 points crossover. It selects the 2 points randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        offspring = numpy.copy(parents)
        crossover_offspring = numpy.uint32(self.crossover_percentage*offspring.shape[0])
        offspring_indices = []
        if self.crossover_percentage != 1:
            offspring_indices = numpy.array(random.sample(range(0, offspring.shape[0]), crossover_offspring))
        else:
            offspring_indices = range(offspring.shape[0])
        
        if (parents.shape[1] == 1): # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
            crossover_point1 = 0
        else:
            crossover_point1 = numpy.random.randint(low=0, high=numpy.ceil(parents.shape[1]/2 + 1), size=1)[0]

        crossover_point2 = crossover_point1 + int(parents.shape[1]/2) # The second point must always be greater than the first point.

        for k in offspring_indices:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]
            # The genes from the beginning of the chromosome up to the first point are copied from the first parent.
            offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
            # The genes from the second point up to the end of the chromosome are copied from the first parent.
            offspring[k, crossover_point2:] = parents[parent1_idx, crossover_point2:]
            # The genes between the 2 points are copied from the second parent.
            offspring[k, crossover_point1:crossover_point2] = parents[parent2_idx, crossover_point1:crossover_point2]
        return offspring

    def uniform_crossover(self, parents, offspring_size):

        """
        Applies the uniform crossover. For each gene, a parent out of the 2 mating parents is selected randomly and the gene is copied from it.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        offspring = numpy.copy(parents)
        
        crossover_offspring = numpy.uint32(self.crossover_percentage*offspring.shape[0])
        offspring_indices = []
        if self.crossover_percentage != 1:
            offspring_indices = numpy.array(random.sample(range(0, offspring.shape[0]), crossover_offspring))
        else:
            offspring_indices = range(offspring.shape[0])

        for k in offspring_indices:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]

            genes_source = numpy.random.randint(low=0, high=2, size=offspring_size[1])
            for gene_idx in range(offspring_size[1]):
                if (genes_source[gene_idx] == 0):
                    # The gene will be copied from the first parent if the current gene index is 0.
                    offspring[k, gene_idx] = parents[parent1_idx, gene_idx]
                elif (genes_source[gene_idx] == 1):
                    # The gene will be copied from the second parent if the current gene index is 1.
                    offspring[k, gene_idx] = parents[parent2_idx, gene_idx]
        return offspring

    def random_mutation(self, offspring):

        """
        Applies the random mutation which changes the values of a number of genes randomly by selecting a random value between random_mutation_min_val and random_mutation_max_val to be added to the selected genes.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
    
        if self.mutation_num_genes == None:
            self.mutation_num_genes = numpy.uint32(self.mutation_percent_genes*offspring.shape[1])
            # Based on the percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
            if self.mutation_num_genes == 0:
                self.mutation_num_genes = 1
        mutation_indices = numpy.array(random.sample(range(0, offspring.shape[1]), self.mutation_num_genes))
        mutated_offspring = numpy.uint32((self.mutation_percentage*offspring.shape[0]))
        offspring_indices = []
        if self.mutation_percentage != 1:      
            offspring_indices = numpy.array(random.sample(range(0, offspring.shape[0]), mutated_offspring))
        else:
            offspring_indices = range(offspring.shape[0])
        
        # Random mutation changes a single gene in each offspring randomly.
        for offspring_idx in offspring_indices:
            for gene_idx in mutation_indices:
                # Generating a random value.
                random_value = numpy.random.randint(self.random_mutation_min_val, self.random_mutation_max_val+1, 1)
                # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                if self.mutation_by_replacement:
                    offspring[offspring_idx, gene_idx] = random_value
                # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                else:
                    offspring[offspring_idx, gene_idx] = offspring[offspring_idx, gene_idx] + random_value
        return offspring

    def swap_mutation(self, offspring):

        """
        Applies the swap mutation which interchanges the values of 2 randomly selected genes.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        mutated_offspring = numpy.uint32(self.mutation_percentage*offspring.shape[0])
        offspring_indices = []
        if self.mutation_percentage != 1:      
            offspring_indices = numpy.array(random.sample(range(0, offspring.shape[0]), mutated_offspring))
        else:
            offspring_indices = range(offspring.shape[0])
        
        for idx in offspring_indices:
            
            mutation_gene1 = numpy.random.randint(low=0, high=offspring.shape[1]/2, size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)

            temp = offspring[idx, mutation_gene1]
            offspring[idx, mutation_gene1] = offspring[idx, mutation_gene2]
            offspring[idx, mutation_gene2] = temp
        return offspring

    def inversion_mutation(self, offspring):

        """
        Applies the inversion mutation which selects a subset of genes and invert them.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        mutated_offspring = numpy.uint32(self.mutation_percentage*offspring.shape[0])
        offspring_indices = []
        if self.mutation_percentage!= 1:      
            offspring_indices = numpy.array(random.sample(range(0, offspring.shape[0]), mutated_offspring))
        else:
            offspring_indices = range(offspring.shape[0])

        for idx in offspring_indices:
            mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)

            genes_to_scramble = numpy.flip(offspring[idx, mutation_gene1:mutation_gene2])
            offspring[idx, mutation_gene1:mutation_gene2] = genes_to_scramble
        return offspring

    def scramble_mutation(self, offspring):

        """
        Applies the scramble mutation which selects a subset of genes and shuffles their order randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        mutated_offspring = numpy.uint32(self.mutation_percentage*offspring.shape[0])
        offspring_indices = []
        if self.mutation_percentage != 1:      
            offspring_indices = numpy.array(random.sample(range(0, offspring.shape[0]), mutated_offspring))
        else:
            offspring_indices = range(offspring.shape[0])

        for idx in offspring_indices:
            mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)
            genes_range = numpy.arange(start=mutation_gene1, stop=mutation_gene2)
            numpy.random.shuffle(genes_range)
            
            genes_to_scramble = numpy.flip(offspring[idx, genes_range])
            offspring[idx, genes_range] = genes_to_scramble
        return offspring

    def best_solution(self):

        """
        Returns information about the best solution found by the genetic algorithm. Can only be called after completing at least 1 generation.
        If no generation is completed (at least 1), an exception is raised. Otherwise, the following is returned:
            -best_solution: Best solution in the current population.
            -best_solution_fitness: Fitness value of the best solution.
            -best_match_idx: Index of the best solution in the current population.
        """
        
        if self.generations_completed < 1:
            raise RuntimeError("The best_solution() method can only be called after completing at least 1 generation but {generations_completed} is completed.".format(generations_completed=self.generations_completed))

#        if self.run_completed == False:
#            raise ValueError("Warning calling the best_solution() method: \nThe run() method is not yet called and thus the GA did not evolve the solutions. Thus, the best solution is retireved from the initial random population without being evolved.\n")

        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        
        
        best_solution_fitness = self.best_solutions_fitness[self.best_solution_generation]
        best_match_idx = self.best_idx[self.best_solution_generation]
        best_solution = self.best_solutions[self.best_solution_generation]

        return best_solution, best_solution_fitness, best_match_idx
        

    def plot_result(self, title="PyGAD - Iteration vs. Fitness", xlabel="Generation", ylabel="Fitness"):

        """
        Creates and shows a plot that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation.
        If no generation is completed, an exception is raised.
        """

        if self.generations_completed < 1:
            raise RuntimeError("The plot_result() method can only be called after completing at least 1 generation but {generations_completed} is completed.".format(generations_completed=self.generations_completed))

#        if self.run_completed == False:
#            print("Warning calling the plot_result() method: \nGA is not executed yet and there are no results to display. Please call the run() method before calling the plot_result() method.\n")

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.best_solutions_fitness)
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.xlabel(xlabel)
        matplotlib.pyplot.ylabel(ylabel)
        matplotlib.pyplot.show()

    def save(self, filename):

        """
        Saves the genetic algorithm instance:
            -filename: Name of the file to save the instance. No extension is needed.
        """

        with open(filename + ".pkl", 'wb') as file:
            pickle.dump(self, file)

def load(filename):

    """
    Reads a saved instance of the genetic algorithm:
        -filename: Name of the file to read the instance. No extension is needed.
    Returns the genetic algorithm instance.
    """

    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Error reading the file {filename}. Please check your inputs.".format(filename=filename))
    except:
        raise BaseException("Error loading the file. Please check if the file exists.")
    return ga_in