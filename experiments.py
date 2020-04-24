# =============================================================================
# A supplementary file to reproduce the following empirical experiments:
# Brute-Force empirical evidence for random strings (experiment 1)
# Boyer-Moore and Boyer-Moore-Horspool empirical comparison (experiment 2)
# =============================================================================

import random # for random number generation
random.seed(123)
import numpy as np
import string # has common ASCII characters
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "cmr10" # setting font
matplotlib.rcParams['font.family'] = "sans-serif"
import matplotlib.transforms as transforms # also for plotting

# Functions needed for both experiments
# =============================================================================

def NumberOfIterations(algorithm, text, list_of_strings, *args):
    '''
    A function to test the number of character comparisons for each word 
    in our list of words called list_of_strings.
        
    Input:
        algorithm - algorithm we are using (as a function name)
        text - the text we are searching in
        list_of_strings - list of words to be searched
        *args - list of additional arguments passed to a function

    Output: 
        a list containing how many iterations it took to find each
        corresponding word in list_of_strings
    '''
    
    n = len(list_of_strings)
    comparisons_arr = np.array(range(0, n))
    
    for i in range(n):
        index, comparisons = algorithm(list_of_strings[i], text, *args)
        comparisons_arr[i] = comparisons
    
    return(comparisons_arr)
    
    
def Naive(pattern, text, leftmost = True):
    '''
    The Brute-Force Algorithm.
    
    Source: 
        directly implemented the pseudo-code given in the project, taken from 
        Robert Sedgewick and Kevin Wayne. Algorithms, 4th Edition.
        Addison-Wesley, 2011

    Input: 
        text - the text we are searching in
        pattern - pattern we are searching for
        leftmost - boolean argument, True if we search only for the first 
                   occurence, False if for all occurences

     Output: 
         a tuple containing the leftmost occurence index (-1 if not found) 
         in the text, and total number of iterations;
         if searching for all words, returns list of indexes and total number
         of iterations for finding all words
    '''
    
    m = len(pattern)
    n = len(text)
    
    iterations = 0    
    index_list = []
    
    if m > n: return -1
    
    # only need to check until m characters till the end
    for i in range(n - m + 1):
        j = 0 # iterates over the needle
          
        while(j < m): 
            iterations += 1
            if (text[i + j] != pattern[j]): break
            j += 1
        
        if (j == m):
            if(leftmost): return [i], iterations
            else: index_list.append(i)
    
    if(len(index_list) == 0): return [-1], iterations
    else: return index_list, iterations

# Experiment 1: Brute-force empirical evidence for random strings
# =============================================================================
    
def CalculateExpectedValue(alphabet, n, m):
    '''
    Finds the exact expected value of comparisons for two random strings
    over given alphabet, as given in Theorem 3.2.
    
    Input: 
        alphabet - alphabet used to construct two random strings
        n - length of the text we are searching in
        m - length of the pattern we are searching for

     Output: 
         expected number of character-to-character comparisons
    '''
    d = len(alphabet)
    return (n - m + 1) * (1 - d**(-m)) / (1 - d**(-1))

def MakeRandomString(length, alphabet):
    ''' 
    Creates a random string of specified length, over a given alphabet.
    
    Input: 
        length - an integer value > 0, the length of the string
        alphabet - alphabet used to construct the random string

     Output: 
        a random string of specified length
    '''
            
    return ''.join(random.choice(alphabet) for i in range(length))

def GeneratePlot(list_iterations, expected_value, x_lab, y_lab, title, 
                 subt):
    
    ''' 
    Makes a scatter plot counting the total number of iterations for each
    of the patterns.
    
    Input: 
        list_iterations - number of iterations it took to find a
        corresponding pattern
        expected_value - expected value, from Theorem 3.2
        x_lab - x-axis label
        y_lab - y-axis label
        title - plot title
        subt - plot subtitle

     Output: 
        produces a plot when we run it
    '''
    
    
    length = len(list_iterations)
    sample_mean = sum(list_iterations) / length
    
    fig, ax = plt.subplots(facecolor="white")
    
    ax.plot(np.array(range(length)), sorted(list_iterations), 
            ls="", marker="o", markersize=0.75, c="orange")
    ax.axhline(expected_value, xmin=0, xmax=length, color="purple", 
               linewidth=0.5, 
               label="E(N) is " + str(round(expected_value,3)))
    ax.axhline(sample_mean, linewidth=0.5,
               xmin=0, xmax=length, color="orange",
               label="Sample mean is " + str(round(sample_mean,3)))
    
    fig.suptitle(title) 
    ax.set_title(subt)
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    ax.legend(loc='lower right')
    return fig
    

def SetupNaive(sigma, n, m, length):
    '''
    Produces experiment 1 setup. First makes a random text of length n and 
    specified number of patterns of length m, both over a given alphabet.
    Then it counts number of iterations it took to find each pattern.

    Input: 
        sigma - alphabet used to construct two random strings
        n - length of the text we are searching in
        m - length of the pattern we are searching for

     Output: 
        a tuple containing a list with the number of iterations it took 
        to find all occurences of each pattern, in the random text of 
        length n, and the expected value of such comparisons
    '''
    random_text = MakeRandomString(n, sigma)
    random_strings = [MakeRandomString(m, sigma) for x in range(length)]
    
    # test how many iterations it took to find all occurences
    random_iterations = NumberOfIterations(Naive, random_text, 
                                           random_strings, False)
    
    return(random_iterations, CalculateExpectedValue(sigma, n, m))


# Reproducing the results for experiment 1
#=============================================================================
# Generate random strings over English lowercase letter alphabet.
    
# We take text of length 100 000, so that n = 100000 and look for
# patterns of length 5, i.e. m = 5. Assume we make a list of 
# 100 such patterns, so that length = 1000.
    
alph_lowercase = string.ascii_lowercase
n, m, length = 100000, 5, 100
random_iterations_lowercase, expected_lowercase = SetupNaive(alph_lowercase, 
                                                             n, m, length)

y_lab = "total number of iterations"
x_lab = "various patterns (sorted by the number of iterations)"
title = "Brute Force Random String Matching"
subtitle_lowercase = "n = 100 000, m = 5, $\sigma=26$"

#------------------------------
# UNCOMMENT TO REPRODUCE PLOTS
#------------------------------
#GeneratePlot(random_iterations_lowercase, expected_lowercase, 
#             x_lab, y_lab, title, subtitle_lowercase)

#plt.savefig('brute_random_english.eps', format='eps', dpi=1000)

# Generate random strings over binary alphabet.
             
alph_binary = '01'
random_iterations_binary, expected_binary = SetupNaive(alph_binary,
                                                       n, m, length)

subtitle_binary = "n = 100 000, m = 5, $\sigma=2$"

#------------------------------
# UNCOMMENT TO REPRODUCE PLOTS
#------------------------------
#GeneratePlot(random_iterations_binary, expected_binary, 
#             x_lab, y_lab, title, subtitle_binary)

#plt.savefig('brute_random_binary.eps', format='eps', dpi=1000)

# Generate random strings over alphabet of all digits, lowercase
# and uppercase english letters and punctuation (94 characters)

alph_all = string.punctuation + string.digits + string.ascii_letters
random_iterations_all, expected_all = SetupNaive(alph_all, n, m, length)

subtitle_all = "n = 100 000, m = 5, $\sigma=94$"

#------------------------------
# UNCOMMENT TO REPRODUCE PLOTS
#------------------------------
#GeneratePlot(random_iterations_all, expected_all, 
#             x_lab, y_lab, title, subtitle_all)

#plt.savefig('brute_random_all.eps', format='eps', dpi=1000)

# Experiment 2: Boyer-Moore VS Boyer-Moore-Horspool testing 
# =============================================================================
def BoyerMooreHorspool(pattern, text):
    '''
    The Boyer-Moore-Horspool Algorithm.
    
    Source: 
        http://code.activestate.com/recipes/117223-boyer-moore-horspool-string-searching/

    Input: 
        text - the text we are searching in
        pattern - pattern we are searching for

     Output: 
         a tuple containing the leftmost occurence index (-1 if not found) 
         in the text, and total number of iterations
    '''
    
    m, n = len(pattern), len(text)
    iterations = 0 # counts number of iterations
    
    if m > n: return -1
    skip = []
    for k in range(256): skip.append(m)
    for k in range(m - 1): skip[ord(pattern[k])] = m - k - 1
        
    skip = tuple(skip)
    k = m - 1
    while k < n:
        j = m - 1; i = k
        
        if text[i] != pattern[j]: iterations += 1
        else:
            while j >= 0 and text[i] == pattern[j]:
                iterations += 1
                j -= 1; i -= 1
                
        if j == -1: return i + 1, iterations 
        k += skip[ord(text[k])]
    return -1, iterations 

'''
Below are some helper functions for the full Boyer-Moore Algotrhom. 
All taken from
https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm
with the function descriptions and comments left as in the original
implementation.
'''

def match_length(S, idx1, idx2):
    '''
    Returns the length of the match of the substrings 
    of S beginning at idx1 and idx2.
    '''
    if idx1 == idx2: return len(S) - idx1
    match_count = 0
    while idx1 < len(S) and idx2 < len(S) and S[idx1] == S[idx2]:
        match_count += 1
        idx1 += 1
        idx2 += 1
    return match_count

def fundamental_preprocess(S):
    '''
    Returns Z, the Fundamental Preprocessing of S. Z[i] is the length of 
    the substring beginning at i which is also a prefix of S. 
    This pre-processing is done in O(n) time, where n is the length of S.
    '''
    
    if len(S) == 0: # Handles case of empty string
        return []
    if len(S) == 1: # Handles case of single-character string
        return [1]
    z = [0 for x in S]
    z[0] = len(S)
    z[1] = match_length(S, 0, 1)
    
    # Optimization from exercise 1-5
    for i in range(2, 1 + z[1]): z[i] = z[1] - i + 1
        
    l, r = 0, 0 # lower and upper limits of z-box
    for i in range(2 + z[1], len(S)):
        if i <= r: # i falls within existing z-box
            k = i - l
            b = z[k]
            a = r - i + 1
            if b < a: # b ends within existing z-box
                z[i] = b
                
            # b ends at or after the end of the z-box, 
            # we need to do an explicit match to the right of the z-box
            else: 
                z[i] = a + match_length(S, a, r + 1)
                l = i 
                r = i + z[i] - 1
        else: # i does not reside within existing z-box
            z[i] = match_length(S, 0, i)
            if z[i] > 0:
                l = i
                r = i + z[i] - 1
    return z

def bad_character_table(S):
    """
    Generates R for S, which is an array indexed by the position of some 
    character c in the English alphabet. At that index in R is an array 
    of length |S|+1, specifying for each index i in S (plus the index after S)
    the next location of character c encountered when traversing S from 
    right to left starting at i. This is used for a constant-time lookup
    for the bad character rule in the Boyer-Moore string search algorithm,
    although it has a much larger size than non-constant-time solutions.
    """
    
    
    if len(S) == 0: return [[] for a in range(256)]
    
    R = [[-1] for a in range(256)]
    alpha = [-1 for a in range(256)]
    for i, c in enumerate(S):
        alpha[ord(c)] = i
        for j, a in enumerate(alpha):
            R[j].append(a)
    return R

def good_suffix_table(S):
    """
    Generates L for S, an array used in the implementation of the strong good 
    suffix rule. L[i] = k, the largest position in S such that S[i:] 
    (the suffix of S starting at i) matches a suffix of S[:k] (a substring 
    in S ending at k). Used in Boyer-Moore, L gives an amount to shift P 
    relative to T such that no instances of P in T are skipped and a 
    suffix of P[:L[i]] matches the substring of T matched by a suffix of P in 
    the previous match attempt. Specifically, if the mismatch took place at
    position i-1 in P, the shift magnitude is given  by the equation 
    len(P) - L[i]. In the case that L[i] = -1, the full shift table is used.
    Since only proper suffixes matter, L[0] = -1.
    """
    L = [-1 for c in S]
    N = fundamental_preprocess(S[::-1]) # S[::-1] reverses S
    N.reverse()
    for j in range(0, len(S)-1):
        i = len(S) - N[j]
        if i != len(S):
            L[i] = j
    return L

def full_shift_table(S):
    """
    Generates F for S, an array used in a special case of the good suffix rule 
    in the Boyer-Moore string search algorithm. F[i] is the length of the 
    longest suffix of S[i:] that is also a prefix of S. In the cases it is
    used, the shift magnitude of the pattern P relative to the text T is 
    len(P) - F[i] for a mismatch occurring at i-1.
    """
    F = [0 for c in S]
    Z = fundamental_preprocess(S)
    longest = 0
    for i, zv in enumerate(reversed(Z)):
        longest = max(zv, longest) if zv == i+1 else longest
        F[-i-1] = longest
    return F

def BoyerMoore(P, T):
    # adapted to return the first match only
    # counts iterations
    
    iterations = 0
    
    """
    Implementation of the Boyer-Moore string search algorithm. This finds 
    all occurrences of P in T, and incorporates numerous ways of 
    pre-processing the pattern to determine the optimal amount to shift the 
    string and skip comparisons. In practice it runs in O(m) (and even 
    sublinear) time, where m is the length of T. This implementation performs 
    a case-insensitive search on ASCII alphabetic characters, spaces 
    not included.
    """
    if len(P) == 0 or len(T) == 0 or len(T) < len(P):
        return -1, iterations

    # Preprocessing
    R = bad_character_table(P)
    L = good_suffix_table(P)
    F = full_shift_table(P)

    k = len(P) - 1      # Represents alignment of end of P relative to T
    previous_k = -1     # Represents alignment in previous phase (Galil's rule)
    while k < len(T):
        i = len(P) - 1  # Character to compare in P
        h = k           # Character to compare in T
        
        # added
        if(P[i] != T[h]):
            iterations += 1
        
        while i >= 0 and h > previous_k and P[i] == T[h]:
            iterations += 1
            i -= 1
            h -= 1
        if i == -1 or h == previous_k:  # Match has been found (Galil's rule)
            matches = k - len(P) + 1
            return (matches, iterations)
        else:  # No match, shift by max of bad character and good suffix rules
            char_shift = i - R[ord(T[h])][i]
            if i+1 == len(P):  # Mismatch happened on first attempt
                suffix_shift = 1
            elif L[i+1] == -1:  # Matched suffix does not appear anywhere in P
                suffix_shift = len(P) - F[i+1]
            else:               # Matched suffix appears in P
                suffix_shift = len(P) - L[i+1]
            shift = max(char_shift, suffix_shift)
            previous_k = k if shift >= i+1 else previous_k  # Galil's rule
            k += shift
    return -1, iterations

def SetupBM(text, list_of_strings):    
    '''
    Produces experiment 2 setup. Given text and a list of patterns, it
    counts number of iterations it took to find each pattern using 
    Boyer-Moore and Boyer-Moore-Horspool algorithms.

    Input: 
        text - the text we are searching in
        list_of_strings - list of words to be searched

     Output: 
        two lists containing the number of iterations it took to find the
        leftmost occurence of each pattern, in the given text, for 
        Boyer-Moore and Boyer-Moore-Horspool algorithms respectively
    '''
    
    # test how many iterations it took to find first occurence:
    iterations_BM = NumberOfIterations(BoyerMoore, text, list_of_strings) #B-M
    iterations_BMH = NumberOfIterations(BoyerMooreHorspool, text,\
                                        list_of_strings) #B-M-H
    
    return(iterations_BM, iterations_BMH)


def GeneratePlotBM(iterations_BM, iterations_BMH, x_lab, y_lab, title, 
                   subt):
    
    ''' 
    Makes a scatter plot counting the total number of iterations for each
    of the patterns. We have two colors for the points - the orange ones
    stand for the Boyer-Moore-Horspool comparisons, and the purple ones are
    for Boyer-Moore.
    
    Input: 
        iterations_BM - number of iterations it took to find a
        corresponding pattern using Boyer-Moore
        iterations_BMH - number of iterations it took to find a
        corresponding pattern using Boyer-Moore-Horspool
        x_lab - x-axis label
        y_lab - y-axis label
        title - plot title
        subt - plot subtitle

     Output: 
        produces a plot when we run it
    '''
    
    length = len(iterations_BM)
    sample_meanBM = sum(iterations_BM)/length
    sample_meanBMH = sum(iterations_BMH)/length
    # sort according to Boyer-Moore-Horspool
    sorted_BMH, sorted_BM = zip(*sorted(zip(iterations_BMH, iterations_BM)))
    
    fig, ax = plt.subplots(facecolor="white")
    
    ax.plot(np.array(range(length)), sorted_BM, 
            ls="", marker="o", markersize=2.5, c="purple")
    
    ax.plot(np.array(range(length)), sorted_BMH, 
            ls="", marker="o", markersize=1.5, c="orange")
    
    ax.axhline(sample_meanBM, xmin=0, xmax=length, color="purple", 
               linewidth=1, label="Sample Mean, Boyer-Moore")
    
    ax.axhline(sample_meanBMH, linewidth=1,
               xmin=0, xmax=length, color="orange",
               label="Sample mean, Boyer-Moore-Horspool")
    
    trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
    
    # add expected value number
    # "{:.0f}".format(expected_value),
    ax.text(.2, sample_meanBM +  2 * sample_meanBM,
             " Sample mean (Boyer Moore) is " + str(round(sample_meanBM,3)), 
             color="purple", transform=trans)
    
    ax.text(.2, sample_meanBMH +  4 * sample_meanBMH,
             " Sample mean (Horspool) is " + str(round(sample_meanBMH,3)), 
             color="orange", transform=trans)
    
    fig.suptitle(title) 
    ax.set_title(subt)
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    ax.legend(loc='upper right')
    return fig
    
# Reproducing the results for experiment 2
#=============================================================================

# Reading in the files
file = open('book-war-and-peace.txt',mode='r')
# read in as one big string
text_book = " ".join(file.read().splitlines())
file.close()

file = open('words.txt',mode='r')
# read in words as a list of words
list_of_patterns_book = np.array(file.read().splitlines())
file.close()

y_lab2 = "total number of iterations"
x_lab2 = "various patterns (sorted by the number of iterations for Horspool)"
title2 = "Boyer-Moore versus Boyer-Moore-Horspool"
subtitle2 = "searching for words in a book "

iterations_BM, iterations_BMH = SetupBM(text_book, list_of_patterns_book)

#------------------------------
# UNCOMMENT TO REPRODUCE PLOTS
#------------------------------
#GeneratePlotBM(iterations_BM, iterations_BMH,
#            x_lab2, y_lab2, title2, subtitle2)

# plt.savefig('boyer_moore.eps', format='eps', dpi=1000)


# checks for the biggest relative differences in pattern searching
BM_smaller = sorted([(iterations_BMH[i], iterations_BM[i],
                   list_of_patterns_book[i]) for i in range(1000)
                  if  iterations_BM[i]<iterations_BMH[i]], 
                    key=lambda t: t[0]/t[1], reverse=True)

BMH_smaller = sorted([(iterations_BM[i], iterations_BMH[i],
                   list_of_patterns_book[i]) for i in range(1000)
                  if  iterations_BMH[i]<iterations_BM[i]],
                     key=lambda t: t[0]/t[1], reverse=True)

equal = [list_of_patterns_book[i] for i in range(1000)
                  if  iterations_BM[i]==iterations_BMH[i]]

