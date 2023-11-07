
"""
# Exercise 2 prompts:

**First**, create a tree with the following format:

```
animals: 
  carnivores: 
    - "cat"
  herbivores: 
    - "koala"
    - "sloth"
  omnivores: 
    - "dog"
    - "human"
```

Hint: You'll have to use lists.

**Second**

Add "bear" to the list of omnivores in `animals` so that the tree looks like

```
animals: 
  carnivores: 
    - "cat"
  herbivores: 
    - "koala"
    - "sloth"
  omnivores: 
    - "dog"
    - "human"
    - "bear"
```

Here you'll use the `set` method introduced above! See [these docs](https://llnl-conduit.readthedocs.io/en/latest/tutorial_python_basics.html) for help.

"""

# conduit + ascent imports
import conduit

# cleanup any old results
!./cleanup.sh

# Create the initial `animals` tree
animals = conduit.Node()
animals["animals/carnivores"] = ["cat"];
animals["animals/herbivores"] = ["koala", "sloth"];
animals["animals/omnivores"] = ["dog", "human"];
print(animals.to_yaml()) 

# Add "bear" to the list of omnivores
animals["animals/omnivores"].append().set("bear")
print(animals.to_yaml()) 
