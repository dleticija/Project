Substring Searching Algorithms - Experiments
--------------------------------------------
This file contains instructions on how to use the file [experiments.py](experiments.py).

Other two files, namely [book text file](book-war-and-peace.txt) and [words text file](words.txt) 
found in the repository are taken from two open GitHub repositories.

Word text file is acquired from [David Norman's GitHub repository](https://gist.github.com/deekayen/4148741), and
book text file is taken from [Luciano Ramalho's GitHub repository](https://github.com/fluentpython/example-code/blob/master/attic/sequences/war-and-peace.txt).

Currently the [experiments file](experiments.py) reproduces the experiments found in 
* Section 3.2 Brute-Force empirical evidence for random strings (experiment 1)
* Section 3.6 Boyer-Moore and Boyer-Moore-Horspool empirical comparison (experiment 2)

Experiment 1
-------------
If you would like to reproduce the plots, scroll down to the section 
"Reproducing the results for experiment 1" in the file, and uncomment the 
following functions:

* for English lowercase alphabet:
  ```python 
  GeneratePlot(random_iterations_lowercase, expected_lowercase, 
             x_lab, y_lab, title, subtitle_lowercase)
   ```
* for binary alphabet:
  ```python 
  GeneratePlot(random_iterations_binary, expected_binary, 
             x_lab, y_lab, title, subtitle_binary)
  ```
* for English lowercase and uppercase letters, digits and ASCII characters:
  ```python 
  GeneratePlot(random_iterations_all, expected_all, 
             x_lab, y_lab, title, subtitle_all)
  ```

To generate your own experiment, you can choose different values for 
text length `n`, pattern length `m`, number of patterns `length` and alphabet `sigma`;

Then run the function 
```python 
SetupNaive(sigma, n, m, length)
``` 
to generate a list containing 
the number of iterations it took to find all occurences of each pattern in 
the random text (call it `random_iterations`), as well as the expected value of such 
comparisons (call it `expected`);

Finally, run 
```python 
GeneratePlot(random_iterations, expected, x_lab, y_lab, title, subtitle)
```
where the other parameters stand for plot aesthetics.

For example, setting `length` to 1000 instead of 100 would result in much closer sample
mean and the expected value curves.


Experiment 2
-------------
To reproduce the plots, scroll down to the section 
"Reproducing the results for experiment 2", and uncomment the following function

```python 
GeneratePlotBM(iterations_BM, iterations_BMH, x_lab2, y_lab2, title2, subtitle2)
```

To generate your own experiment, you can upload your own text file `text` and 
your own list of patterns `list_of_patterns`;

Then run 
```python 
SetupBM(text, list_of_patterns)
```
to generate a list containing 
the number of iterations it took to find all occurences of each pattern in 
the given text using Boyer-Moore Algorithm (`iterations_BM`) and 
Boyer-Moore-Horspool Algorithm (`iterations_BMH`);

Finally, run 
```python 
GeneratePlotBM(iterations_BM, iterations_BMH, x_lab2, y_lab2, title2, subtitle2)
``` 
where the other parameters stand for plot aesthetics.




