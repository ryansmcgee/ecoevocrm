import numpy as np
import scipy
#------------------------------
#import portal.utils as utils
#import portal.matrices as matrices

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Landscape:

    def __init__(self, genotype_fitnesses=None, genotypes=None, N=None, fitnesses=None):

        self.N = N
        
        if(genotype_fitnesses is not None and isinstance(genotype_fitnesses, dict)):
            self.genotype_fitnesses = genotype_fitnesses
            self.genotypes = np.array(list(self.genotype_fitnesses.keys()))
            self.fitnesses = np.array(list(self.genotype_fitnesses.values()))
            self.num_genotypes = len(self.genotypes)

        elif(fitnesses is not None):
            self.fitnesses = np.array(fitnesses)
            if(genotypes is not None):
                self.genotypes = genotypes
            elif(N is not None):
                self.genotypes = [bin(i)[2:].zfill(N) for i in range(2**N)]
            else:
                utils.error("Either a list of genotypes or a number of loci (N) must be provided.")
            self.num_genotypes = len(self.genotypes)
            assert(self.num_genotypes == len(fitnesses)), f"Number of fitness values ({len(fitnesses)}) must match the number of genotypes ({self.num_genotypes})."
            self.genotype_fitnesses = dict(zip(self.genotypes, self.fitnesses))

        else:
            utils.error("Either a dictionary of {genotype: fitness} pairs or lists of genotypes and fitnesses must be provided.")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_genotype_fitness(self, genotype):
        return self.genotype_fitnesses[genotype]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class NKLandscape:

    def __init__(self, N, K, locus_interactions='random', seed=None):

        self.seed = np.random.randint(low=0, high=1e9) if seed is None else seed
        np.random.seed(self.seed)

        self.N  = N
        self.K  = K
        if(self.K > self.N-1):
            raise Exception("K must be <= N-1")

        # Generate all binary strings with N digits:
        self.genotypes     = [bin(i)[2:].zfill(self.N) for i in range(2**self.N)]
        self.num_genotypes = len(self.genotypes)

        # If an interactions matrix is provided, use the given matrix:
        if(type(locus_interactions) is np.ndarray):
            if(len(locus_interactions.shape)==2 and locus_interactions.shape[0] == N and locus_interactions.shape[1] == K):
                self.locusInteractions = locus_interactions
            else:
                raise Exception("The dimensions of a given locus interactions matrix should be NxK. The expected dimensions are "+str(self.N)+"x"+str(self.K))
        # Otherwise, generate a new interactions matrix using the given interaction scheme (e.g., random, adjacent, etc):
        else:
            self.locusInteractions = self.generate_locus_interaction_matrix(locus_interactions)

        # (locus, epistaticState): fitnessContribution
        #    Ex: (2, -01-0-): 0.87
        #    This lookup table (dict) will be populated in the process of generate_landscape().
        self.lociEpistaticFitnessesTable = {}

        # genotype: fitness
        #    Ex: '001101': 0.42
        #    This lookup table (dict) will be populated in the process of generate_landscape().
        self.genotype_fitnesses = {}

        self.generate_landscape()

        self.fitnesses = np.array(list(self.genotype_fitnesses.values()))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # @property
    # def fitnesses(self):
    #   return np.array(list(self.genotype_fitnesses.values()))

    # @property
    # def num_genotypes(self):
    #   return len(self.genotypes)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_locus_interaction_matrix(self, interaction_scheme):

        locusInteractionMatrix = np.zeros(shape=(self.N, self.N))

        if(interaction_scheme == 'random'):

            for focalLocus in range(self.N):

                # This locus interacts with itself (always) and K other RANDOMLY CHOSEN loci:
                otherLoci = [i for i in range(self.N) if i != focalLocus]
                interactingLoci = np.append(focalLocus, np.random.choice(otherLoci, size=self.K, replace=False)).astype(np.int16)

                focalLocusInteractions = np.zeros(self.N)
                focalLocusInteractions[interactingLoci] = 1

                locusInteractionMatrix[focalLocus] = focalLocusInteractions

        elif(interaction_scheme == 'adjacent'):
            
            for focalLocus in range(self.N):

                # This locus interacts with itself (always) and K other ADJACENT loci:
                # (The K adjacent loci are split evenly on both sides of the focal locus)
                # (Adjacency is cyclic, i.e., the Nth locus is adjacent to the 0th locus)
                interactingLoci = np.mod(range(focalLocus-int(self.K/2), focalLocus-int(self.K/2)+self.K+1), self.N)

                focalLocusInteractions = np.zeros(self.N)
                focalLocusInteractions[interactingLoci] = 1

                locusInteractionMatrix[focalLocus] = focalLocusInteractions

        return locusInteractionMatrix

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def generate_landscape(self):

        for genotype in self.genotypes:
            self.genotype_fitnesses[genotype] = self.calc_genotype_fitness(genotype)

        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def calc_genotype_fitness(self, genotype):

        lociFitnessContributions = []

        for focalLocus in range(self.N):
            lociFitnessContributions.append(self.calc_locus_fitness_contribution(genotype, focalLocus))

        genotypeFitness = np.mean(lociFitnessContributions)

        return genotypeFitness

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def calc_locus_fitness_contribution(self, genotype, locus):

        locusInteractions = self.locusInteractions[locus]
        epistaticState = ''.join([locusState if locusInteractions[i]==1 else '-' for i, locusState in enumerate(genotype)])

        locusEpistaticState = tuple((locus, epistaticState))

        try:
            locusFitnessContribution = self.lociEpistaticFitnessesTable[ locusEpistaticState ]
        except KeyError:
            self.lociEpistaticFitnessesTable[ locusEpistaticState ] = np.random.uniform(0, 1)
            locusFitnessContribution = self.lociEpistaticFitnessesTable[ locusEpistaticState ]

        return locusFitnessContribution

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_genotype_fitness(self, genotype):
        return self.genotype_fitnesses[genotype]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_fitness_vector(self):
        fitnessVector = np.array([self.genotype_fitnesses[g] for g in self.genotypes])
        
        return fitnessVector


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def merge_landscapes(dict_A, dict_B, prefix_A='', prefix_B='', suffix_A='', suffix_B=''):
    dict_A = {f'{prefix_A}{key}{suffix_A}': fitness for key, fitness in dict_A.items()}
    dict_B = {f'{prefix_B}{key}{suffix_B}': fitness for key, fitness in dict_B.items()}
    merged_dict = {**dict_A, **dict_B} # this merges the two dicts (B overwrites A if overlapping keys)
    return merged_dict

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def mutsel_equilibrium(landscape, mutation_rate, max_mutation_step=1):
    # ~ ~ ~ ~
    # Reference:
    # Hermisson, Joachim, et al. "Mutation–selection balance: ancestry, load, and maximum principle."
    # Theoretical population biology 62.1 (2002): 9-46.
    # ~ ~ ~ ~

    R = np.diag(landscape.fitnesses)
    R_mean = np.identity(R.shape[0]) * np.mean(landscape.fitnesses)
    
    U = matrices.generate_mutation_rate_matrix(landscapes=landscape, mutation_rates=mutation_rate, max_mutation_step=max_mutation_step, include_null_genotype=False)
    U = U.toarray()
    
    M_influx = landscape.fitnesses * U    # M_in_ij = B_j * u_ij
    np.fill_diagonal(M_influx, 0)
    
    M_outflux = landscape.fitnesses * U.T         # Mout_ij = B_j * u_ji
    np.fill_diagonal(M_outflux, 0)
    M_outflux_diag = -1 * np.sum(M_outflux, axis=0) # sum columns :: M_out_ii = sum_{i!=j}[B_i * u_ji] (where i!=j)
    M_outflux = np.zeros_like(M_outflux)
    np.fill_diagonal(M_outflux, M_outflux_diag)

    M = M_influx + M_outflux
    
    H = R + M
    
    #-----
    
    eigvals, eigvecs = np.linalg.eig(H)
    
    p_eq = eigvecs[:, np.argmax(eigvals)]/eigvecs[:, np.argmax(eigvals)].sum()
    
    return p_eq


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MI_similarity(landscapes, mutation_rate, q_landscapes=None, return_similarity_score=True):
    
    numG = len(landscapes[0].genotypes)
    numL = len(landscapes)
    
    pq_joint = np.zeros((numG, numL))
    
    for l in range(numL):
        pq_joint[:,l] = mutsel_equilibrium(landscapes[l], mutation_rate)
        
    q_landscapes = np.array([1/numL]*numL) if q_landscapes is None else q_landscapes
    
    pq_joint = pq_joint * q_landscapes
    
    H_E   = scipy.stats.entropy(pq_joint.sum(axis=0), base=2)
    H_F   = scipy.stats.entropy(pq_joint.sum(axis=1), base=2)
    H_EF  = scipy.stats.entropy(pq_joint.flatten(), base=2)
    MI_EF = H_E + H_F - H_EF
    
    return (H_E - MI_EF) if return_similarity_score else MI_EF


