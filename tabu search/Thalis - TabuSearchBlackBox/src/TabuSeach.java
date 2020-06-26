import java.io.File;
import java.util.*;

public class TabuSeach {

    //parameters
    private static int length;
    private static int popSize = 10;
    private static int gamma = 10000;
    private static int a = 1000;
    private static String filepath = "tests/set0a/n0000012i00.txt";

    private static int[][] weights;
    private static int goal;
    private static long numOfEvals = 0;
    private static long generations = 0;
    private static long startTime;

    private static boolean[] tabu_search(boolean[] x_old) {

        int best_solution_fitness = Integer.MIN_VALUE;
        boolean[] best_solution = x_old;

        boolean[] x = Arrays.copyOf(x_old, length);

        int[] d = new int[length];
        Random r = new Random();
        int tl;
        int bound = (int) (0.15 * length)-2;
        if (bound <= 0)
            tl = length/6;
        else tl = r.nextInt(bound) + 3;
        int iter = 0;
        int[] TL;
        //Queue<Integer> triedMovesFlip = new LinkedList<>();
        //Queue<Integer> triedMovesExchange = new LinkedList<>();

        int fit = fitness(x);

        //initialize d
        for (int i = 0; i < length; i++) {
            if (!x[i]) {
                for (int j = 0; j < length; j++) {
                    if (!x[j]) {
                        d[i] += weights[i][j];
                    } else d[i] -= weights[i][j];
                }
            } else {
                for (int j = 0; j < length; j++) {
                    if (!x[j]) {
                        d[i] -= weights[i][j];
                    } else d[i] += weights[i][j];
                }
            }
        }

        while (iter < gamma) {

            int nonImpN1 = 0;
            int nonImpN2 = 0;
            TL = new int[length];



            do {

                //System.out.println(Arrays.toString(d));

                //identify best flip
                int best_gain = Integer.MIN_VALUE;
                int index = -1;
                int new_fit = Integer.MIN_VALUE;
                for (int j = 0; j < length; j++) {
                   // if (d[j] > best_gain && (!triedMovesFlip.contains(j) || fit + d[j] > best_solution_fitness)) {
                    if (d[j] > best_gain && (iter > TL[j] || fit + d[j] > best_solution_fitness)) {
                    best_gain = d[j];
                        index = j;
                        new_fit = fit + best_gain;
                    }
                }

                //update list of tried moves
                /*
                if (triedMovesFlip.size() >= tl)
                    triedMovesFlip.remove();
                triedMovesFlip.add(index);
                 */


                x[index] = !x[index];

                //update d
                for (int k = 0; k < length; k++) {
                    if (index == k) {
                        d[k] = -d[k];
                    } else if (x[k] != x[index]) {
                        d[k] -= 2 * weights[index][k];
                    } else {
                        d[k] += 2 * weights[index][k];
                    }
                }

                TL[index] = iter + tl;

                if (new_fit > best_solution_fitness) {
                    best_solution_fitness = new_fit;
                    best_solution = Arrays.copyOf(x, length);
                    nonImpN1 = 0;
                } else nonImpN1++;

                iter = iter + 1;
                numOfEvals++;
                fit = new_fit;

            } while (nonImpN1 < a);



            do {

                //identify best exchange
                if (fit == 0) x[0] = !x[0];

                int best_gain_i = Integer.MIN_VALUE;
                int best_gain_j = Integer.MIN_VALUE;
                int index_i = -1;
                int index_j = -1;

                for (int i=0; i< length; i++) {
                    //if (!triedMovesExchange.contains(i)){
                    if (iter > TL[i]){
                        if (x[i]){
                            if (d[i] > best_gain_i){
                                best_gain_i = d[i];
                                index_i = i;
                            }
                        } else {
                            if (d[i] > best_gain_j){
                                best_gain_j = d[i];
                                index_j = i;
                            }
                        }
                    }
                }

                int new_fit = fit + d[index_i] + d[index_j] + 2*weights[index_i][index_j];


                //update list of tried moves
                /*
                if (triedMovesExchange.size() >= 2*tl) {
                    triedMovesExchange.remove();
                    triedMovesExchange.remove();
                }
                triedMovesExchange.add(index_i);
                triedMovesExchange.add(index_j);

                 */


                x[index_i] = !x[index_i];
                x[index_j] = !x[index_j];

                //update d
                for (int k = 0; k < length; k++) {
                    if (index_i == k || index_j == k) {
                        d[k] = -d[k] - 2 * weights[index_i][index_j];
                    } else if (x[k] != x[index_i]) {
                        d[k] += -2 * weights[index_i][k] + 2 * weights[index_j][k];
                    } else {
                        d[k] += +2 * weights[index_i][k] - 2 * weights[index_j][k];
                    }
                }

                TL[index_i] = iter + tl;
                TL[index_j] = iter + tl;

                if (new_fit > best_solution_fitness) {
                    best_solution_fitness = new_fit;
                    best_solution = Arrays.copyOf(x, length);
                    nonImpN2 = 0;
                } else nonImpN2++;

                iter = iter + 2;
                numOfEvals++;
                fit = new_fit;

            } while (nonImpN2 < a);


        } // end of iter while

        return best_solution;

    }

    private static List<boolean[]> createInitialPopulation() {
        Random r = new Random();
        List<boolean[]> population = new LinkedList<>();
        //produce 3*popSize solutions
        for (int i = 0; i < 3 * popSize; i++) {
            //create random solution
            boolean[] x = new boolean[length];
            for (int j = 0; j < length; j++) {
                if (r.nextDouble() >= 0.5)
                    x[j] = true;
            }

            //perform tabu search
            x = naiveTabu_search(x);

            //check if solution already exists
            boolean exists = false;
            for (boolean[] sol : population) {
                if (Arrays.equals(x, sol)) {
                    exists = true;
                    break;
                }
                int k;
                for (k = 0; k < length; k++) {
                    if (x[k] == sol[k]) break;
                }
                if (k == length) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                population.add(x);
                int fit = fitness(x);
                System.out.println(i + ": " + fit);
                if (fit == goal){
                    long endTime = System.currentTimeMillis();
                    System.out.println("optimal solution fitness: " + fit);
                    System.out.println("optimal solution: " + Arrays.toString(x));
                    System.out.println("NumberOfEvals: " + numOfEvals);
                    System.out.println("NumberOfGenerations: " + generations);
                    System.out.println("Time: " + (endTime - startTime) + " ms");
                    System.exit(0);


                    //System.out.println((endTime - startTime) + "\t" + fit + "\t" + numOfEvals + "\t1\t" + generations);
                    //return null;

                }
            }
            else i--;
        }

        //select top popSize solutions
        population.sort((x1, x2) -> {
            int fit1 = fitness(x1);
            int fit2 = fitness(x2);
            return Integer.compare(fit2, fit1);
        });

        population = population.subList(0, popSize);

        return population;
    }

    private static int fitness(boolean[] x) {
        int total = 0;
        for (int i = 0; i < length; i++) {
            int n1 = x[i] ? 1 : 0;
            for (int j = 0; j < length; j++) {
                int n2 = x[j] ? 1 : 0;
                if (i != j) {
                    total += weights[i][j] * (1 - n1) * n2;
                }
            }
        }
        return total;

    }

    private static int averagePopulationDistance(List<boolean[]> population) {
        int total = 0;
        for (int i = 0; i < popSize; i++) {
            for (int j = i + 1; j < popSize; j++) {
                total += distance(population.get(i), population.get(j));
            }
        }
        return (int) (total * 2.0 / (popSize * (popSize - 1)));
    }

    private static int distance(boolean[] x1, boolean[] x2) {
        int HD = 0;
        for (int i = 0; i < length; i++) {
            if (x1[i] != x2[i])
                HD++;
        }
        if (HD > length / 2) {
            HD = length - HD;
            for (int i = 0; i < length; i++) {
                x1[i] = !x1[i];
            }
        }
        return HD;
    }

    private static boolean[] combinationOperator(boolean[] x1, boolean[] x2) {

        boolean[] offspring = new boolean[length];

        //initialize index lists C and NC
        List<Integer> C = new LinkedList<>();
        List<Integer> NC = new LinkedList<>();
        for (int i = 0; i < length; i++) {
            if (x1[i] == x2[i])
                C.add(i);
            else NC.add(i);
        }
        if (C.size() < length / 2) {
            C.clear();
            NC.clear();
            for (int i = 0; i < length; i++) {
                if (x1[i] != x2[i])
                    C.add(i);
                else NC.add(i);
            }
        }

        int NClength = NC.size();

        //initialize fixed part of offspring
        for (int i = 0; i < length; i++) {
            if (C.contains(i))
                offspring[i] = x1[i];
        }

        //initialize g0/g1 contribution lists
        int[] g0 = new int[length];
        int[] g1 = new int[length];
        for (int i : NC) {
            int total1 = 0;
            int total2 = 0;
            for (int j : C) {
                if (offspring[j])
                    total1 += weights[i][j];
                else total2 += weights[i][j];
            }
            g0[i] = total1;
            g1[i] = total2;
        }

        //repeat until you fill up the offspring
        for (int k = 1; k <= NClength; k++) {
            boolean[] p;
            if (k % 2 == 0)
                p = x1;
            else p = x2;
            //find maximum-contribution variable
            int max_contr = Integer.MIN_VALUE;
            int index = -1;
            for (int ind : NC) {
                if (p[ind]) {
                    if (g1[ind] > max_contr) {
                        max_contr = g1[ind];
                        index = ind;
                    }
                } else {
                    if (g0[ind] > max_contr) {
                        max_contr = g0[ind];
                        index = ind;
                    }
                }
            }

            //update offspring
            offspring[index] = p[index];
            //delete from NC
            for (int i = 0; i < NC.size(); i++) {
                if (NC.get(i) == index) {
                    NC.remove(i);
                    break;
                }
            }

            //update g0/g1 contribution lists
            if (p[index]) {
                for (int ind : NC) {
                    g0[ind] += weights[ind][index];
                }
            } else {
                for (int ind : NC) {
                    g1[ind] += weights[ind][index];
                }
            }

            numOfEvals++;

        } //end of filling up offspring

        return offspring;

    }

    private static boolean[] naiveCombinationOperator(boolean[] x1, boolean[] x2) {

        boolean[] offspring = new boolean[length];
        Random r = new Random();

        for (int i = 0; i < length; i++) {
            if (r.nextDouble() >= 0.5)
                offspring[i] = x2[i];
             else offspring[i] = x1[i];
        }

        return offspring;
    }

    private static boolean[] naiveTabu_search(boolean[] x_old) {

        int best_solution_fitness = Integer.MIN_VALUE;
        boolean[] best_solution = x_old;

        boolean[] x = Arrays.copyOf(x_old, length);

        int[] d = new int[length];
        Random r = new Random();
        int tl;
        int bound = (int) (0.15 * length)-2;
        if (bound <= 0)
            tl = length/6;
        else tl = r.nextInt(bound) + 3;
        int iter = 0;
        int[] TL;
        //Queue<Integer> triedMovesFlip = new LinkedList<>();
        //Queue<Integer> triedMovesExchange = new LinkedList<>();

        int fit = fitness(x);

        //initialize d
        for (int i = 0; i < length; i++) {
            if (!x[i]) {
                for (int j = 0; j < length; j++) {
                    if (!x[j]) {
                        d[i] += weights[i][j];
                    } else d[i] -= weights[i][j];
                }
            } else {
                for (int j = 0; j < length; j++) {
                    if (!x[j]) {
                        d[i] -= weights[i][j];
                    } else d[i] += weights[i][j];
                }
            }
        }

        while (iter < gamma) {

            int nonImpN1 = 0;
            int nonImpN2 = 0;
            TL = new int[length];


            do {

                //System.out.println(Arrays.toString(d));

                //identify best flip
                int index;
                int c=0;

                do {
                    index = r.nextInt(length);
                    c++;
                    if (c > 100) break;
                } while (iter <= TL[index] && fit + d[index] <= best_solution_fitness);

                int best_gain = d[index];
                int new_fit = fit + best_gain;

                x[index] = !x[index];

                //update d
                for (int k = 0; k < length; k++) {
                    if (index == k) {
                        d[k] = -d[k];
                    } else if (x[k] != x[index]) {
                        d[k] -= 2 * weights[index][k];
                    } else {
                        d[k] += 2 * weights[index][k];
                    }
                }

                TL[index] = iter + tl;

                if (new_fit > best_solution_fitness) {
                    best_solution_fitness = new_fit;
                    best_solution = Arrays.copyOf(x, length);
                    nonImpN1 = 0;
                } else nonImpN1++;

                iter = iter + 1;
                numOfEvals++;
                fit = new_fit;

            } while (nonImpN1 < a);



            do {

                //identify best exchange

                if (fit == 0) x[0] = !x[0];

                int index_i;
                int index_j;
                int c=0;

                do {
                    index_i = r.nextInt(length);
                    index_j = r.nextInt(length);
                    c++;
                    if (c > 100) break;
                } while ( (iter <= TL[index_i] || iter <= TL[index_j] || x[index_i] == x[index_j]) &&
                        fit + d[index_i] + d[index_j] + 2*weights[index_i][index_j] <= best_solution_fitness);


                int new_fit = fit + d[index_i] + d[index_j] + 2*weights[index_i][index_j];


                x[index_i] = !x[index_i];
                x[index_j] = !x[index_j];

                //update d
                for (int k = 0; k < length; k++) {
                    if (index_i == k || index_j == k) {
                        d[k] = -d[k] - 2 * weights[index_i][index_j];
                    } else if (x[k] != x[index_i]) {
                        d[k] += -2 * weights[index_i][k] + 2 * weights[index_j][k];
                    } else {
                        d[k] += +2 * weights[index_i][k] - 2 * weights[index_j][k];
                    }
                }

                TL[index_i] = iter + tl;
                TL[index_j] = iter + tl;

                if (new_fit > best_solution_fitness) {
                    best_solution_fitness = new_fit;
                    best_solution = Arrays.copyOf(x, length);
                    nonImpN2 = 0;
                } else nonImpN2++;

                iter = iter + 2;
                numOfEvals++;
                fit = new_fit;

            } while (nonImpN2 < a);


        } // end of iter while

        return best_solution;

    }

    public static void main(String[] args) {

        Random r = new Random();

        File file = new File(filepath);
        try {
            Scanner sc = new Scanner(file);
            length = sc.nextInt();
            weights = new int[length][length];
            int numEdges = sc.nextInt();
            for (int i = 0; i < numEdges; i++) {
                int u = sc.nextInt() - 1;
                int v = sc.nextInt() - 1;
                weights[u][v] = weights[v][u] = sc.nextInt();
            }
            goal = sc.nextInt();
            sc.close();
        } catch (Exception e) {
            System.out.println("File not found");
            return;
        }

        /*
        for (int i=0; i<length; i++){
            for (int j=i+1; j<length; j++){
                weights[i][j] = weights[j][i] = r.nextInt(20);
            }
        }
         */


        /*
        weights[0][1] = weights[1][0] = 5;
        weights[0][2] = weights[2][0] = 10;
        weights[1][2] = weights[2][1] = 6;

         */

        //for (int round=0; round<11; round++) {

        numOfEvals = 0;
        generations = 0;
        startTime = System.currentTimeMillis();

        System.out.println("Adding instances to initial population...");
        List<boolean[]> population = createInitialPopulation();
        //if (population == null) continue;

        //record best solution and its fitness
        int best_solution_fitness_gl = fitness(population.get(0));
        boolean[] best_solution_gl = Arrays.copyOf(population.get(0), length);
        System.out.println("best fitness in population:" + best_solution_fitness_gl);

        //add new solution/offspring to the population
        do {

            generations++;

            //select parents
            int averagePopDistance = averagePopulationDistance(population);
            boolean[] p1;
            boolean[] p2;
            do {
                p1 = population.get(r.nextInt(popSize));
                p2 = population.get(r.nextInt(popSize));
            } while (distance(p1, p2) <= averagePopDistance);

        boolean[] offspring = naiveCombinationOperator(p1, p2);
        boolean[] x = naiveTabu_search(offspring);

            //POPULATION UPDATE RULE
            //check if solution already exists in population
            boolean exists = false;
            for (boolean[] sol : population) {
                if (Arrays.equals(x, sol)) {
                    exists = true;
                    break;
                }
                int k;
                for (k = 0; k < length; k++) {
                    if (x[k] == sol[k]) break;
                }
                if (k == length) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                //check if solution is better than at least one in population
                int fit = fitness(x);
                if (fitness(population.get(population.size() - 1)) >= fit)
                    continue;
                int i;
                for (i = population.size() - 2; i >= 0; i--) {
                    if (fitness(population.get(i)) >= fit) {
                        //push all elements in population one seat to the right
                        boolean[] sol = population.set(i + 1, x);
                        for (int j = i + 2; j < population.size(); j++)
                            sol = population.set(j, sol);
                        break;
                    }
                }
                //if you are better than everybody else...
                if (i < 0) {
                    System.out.println("Best solution fitness switched from " + best_solution_fitness_gl + " to " + fit);
                    best_solution_fitness_gl = fit;
                    best_solution_gl = Arrays.copyOf(x, length);
                    boolean[] sol = population.set(0, x);
                    for (int j = 1; j < population.size(); j++)
                        sol = population.set(j, sol);
                }
            }

            long time = System.currentTimeMillis();
            if (time - startTime > 180000){     //3minutes
                System.out.println("best solution fitness found: " + best_solution_fitness_gl);
                System.out.println("best solution found: " + Arrays.toString(best_solution_gl));
                System.out.println("NumberOfEvals: " + numOfEvals);
                System.out.println("NumberOfGenerations: " + generations);
                System.out.println("Timeout: " + (time - startTime) + " ms");
                System.exit(0);



                //System.out.println((time - startTime) + "\t" + best_solution_fitness_gl + "\t" + numOfEvals + "\t1\t" + generations);
                //break;

            }


        } while (best_solution_fitness_gl < goal);

        //if (best_solution_fitness_gl < goal)
            //continue;

        long endTime = System.currentTimeMillis();

        System.out.println("optimal solution fitness: " + best_solution_fitness_gl);
        System.out.println("optimal solution: " + Arrays.toString(best_solution_gl));
        System.out.println("NumberOfEvals: " + numOfEvals);
        System.out.println("NumberOfGenerations: " + generations);
        System.out.println("Time: " + (endTime - startTime) + " ms");


        //System.out.println((endTime - startTime) + "\t" + best_solution_fitness_gl + "\t" + numOfEvals + "\t1\t" + generations);

        //}

    }

}
