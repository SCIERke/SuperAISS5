# Python OOP

# What is OOP?

OOP  ( Object-Oriented Programming ) is type of programming method that help us to write code clean, organize and robust.

OK, Let’s dive into python OOP!!

 In python, we can create class by simply writing this code.

```python
class Human:
	x = 2
	y = 4 # here is private valuable
```

here is how can we can access our valuable!

```python
Human.x # 2 / access by class directly

# or we can access it by create class object!
peerapat = Human()
peerapat.x 
```

# Conventional Class Format

Conventional Class Format in python is necessary thing you need to know! because when we work with many people, we need to quality code not just work code!

Convention Coding help us to format our code to easy to read ,robust and everyone write the same way as your code. Following conventions may not be necessary to make the compiler work, but it helps us collaborate with other people's work!

Let’s get back to code! here is how we create conventional class in python:

```python
class ClassName:
    def __init__(self, arg1, arg2):       # Constructor method
        self.arg1 = arg1                  # Instance attribute
        self.arg2 = arg2

    def _private_method(self):            # Instance method
        return self.arg1

    def public_method(self, x):           # Method with extra argument
        return self.arg2 + x
```

from the above code , you may curious what is `self` argument do? why it need to be pass in every parameter of function?!

The `self` is used **to represent the instance of the class**. With this keyword, you can access the attributes and methods of the class in python. It binds the attributes with the given arguments.

## Class Constructor

In class-based, object-oriented programming, a constructor (abbreviation: ctor) is **a special type of function called to create an object**. It prepares the new object for use, often accepting arguments that the constructor uses to set required member variables.

Class Constructor in python may look a bit different from other language because in python when we create instance , python will do 2 steps.

- **First**, Python automatically calls `Dog.__new__(cls)` to create **an empty Dog object**.
    
    (`cls` is the class itself, here `Dog`)
    
- **Then**, Python **calls your** `Dog.__init__(self, "Buddy")` to **initialize** that object.

Now you may curious what the fuck that `__new__` and `__init__` do?

In Python, **`__init__`** is **the initializer** — it **acts like** a **constructor** for a class.

But **technically**, if you want to be super correct:

- **`__new__`** is the **real constructor** (it *creates* the object in memory).
- **`__init__`** is the **initializer** (it *sets up* the object after creation).

| Step | Method | What it does |
| --- | --- | --- |
| Step 1 | `__new__` | Allocates memory, creates a blank object |
| Step 2 | `__init__` | Initializes the object's attributes (like setting `self.name = name`) |

In normal coding, you **only write `__init__`**, because Python handles `__new__` for you automatically unless you need very fancy stuff. **And 99% of the time, when people say "constructor" in Python, they actually mean `__init__`. So you should understand it this way, even though it's not technically a fact.**

OK now we all understand Class Constructor , Conventional Class Format Concept. Let’s get back to coding!!!

For example, When i want to create Mammals class, I may be write code like this:

```python
class Mammals:
    def __init__(self, name:str, n_legs:int, sound:str):
        self.name = name
        self.n_legs = n_legs
        self.sound = sound
        self._greeting()

    def _greeting(self):
        print(f"{self.sound}!, Hello {self.name}")

    def speak(self):
        print(f"{self.name} sound is {self.sound}")

    def count_legs(self, n_mammals:int):
        n_legs = self.n_legs * n_mammals
        print(f"{n_mammals} {self.name} have {n_legs} legs")
        return n_legs
```

And when we want to create object. we can do like this:

```python
senmee = Mammals(name="Senmee", n_legs=2, sound="SaWadDee")
```

## **Object vs Instance:**

- **Object**
    - An **object** is simply a **block of memory** that holds data and methods (functions).
    - It's created when a class is instantiated.
    - In Python, when you create an object, you’re essentially creating an instance of a class, but **the term "object" is more general**. It can refer to any entity created from a class.
- **Instance**
    - An **instance** is a **specific occurrence** of a class (an object created from that class).
    - In simpler terms, the term **instance** specifically refers to an **object** of a particular class.
    - An **instance** means "an object that is an occurrence of a class," or in other words, **it is a concrete representation** of the class, holding the actual data of that class.

### **Quick analogy**:

- Imagine a **class** as a **blueprint** (like a dog breed blueprint).
- The **object** is the **dog** created using that blueprint (the actual dog).
- Each **dog** (object) created from the blueprint (class) is an **instance** of the dog breed

## Methods

When you define functions **inside a class** like you did (for example, _greeting, speak, count_legs),
those are simply called **methods.**

| What | Name |
| --- | --- |
| Function inside a class | **Method** |
| Method that needs `self` (normal method) | **Instance method** |
| Method with `@classmethod` | **Class method** |
| Method with `@staticmethod` | **Static method** |

### Instance Method

An instance method is a method that works with a specific instance (or object) of a class.
It operates on the data stored in that object (using self to access it).

```python
def _greeting(self):
def speak(self):
def count_legs(self, n_mammals: int):
```

They are **instance methods**, because they use `self` to work on **that specific object**.

### Class Method

A **class method** is a method that **works with the class itself**, **not with an object**. It **receives the class** as its first argument (usually called `cls`), instead of `self`. You create a class method using `@classmethod` decorator

**In normal methods**:

- `self` → means "this specific object (instance)"
    
    (Example: "this specific dog, not all dogs")
    

**In class methods**:

- `cls` → means "the class itself"
    
    (Example: "the Dog *class* as a whole")
    

```python
class Dog:
    species = "Canine"  # class variable

    def __init__(self, name):
        self.name = name

    @classmethod
    def describe_species(cls):
        print(f"All dogs are {cls.species}")

# Using
Dog.describe_species()   # ✅ No need to create object first!
```

See? `describe_species` is a **class method** that talks about the **whole Dog class**, not about one dog.

You may something like this

```python
class Dog:
    def __init__(self, name):
        self.name = name

    @classmethod
    def from_breed(cls, breed):
        return cls(name=f"A {breed} dog")

dog = Dog("Buddy")              # normal way
dog2 = Dog.from_breed("Husky")  # alternative way using @classmethod
```

And you may have something in your heart. Why does `from_breed` look like an instance method?

**Key Differences:**

- **Instance Method**: You call it **on an object** and it operates on that object (using `self`).
- **Class Method**: You call it **on the class itself** and it operates on the class (using `cls`), but it **can also return an object** of the class, just like the constructor (`__init__`).

**`from_breed`** is **returning an object** (like `__init__` does), it may *feel* like it's behaving the same as an instance method. The difference is that you’re **calling it on the class** (using `Dog.from_breed(...)`), not on an instance.

### Static Methods

A static method is a method inside a class that does not rely on any instance or class-specific data. It behaves like a regular function but belongs to the class’s namespace. It doesn't take self or cls as its first argument.

## Python Inheritance

Inheritance is the mechanism by which **one class is allowed to inherit the features(fields and methods) of another class.**

**A class that inherits from another class can reuse the methods and fields of that class. In addition, you can add new fields and methods to your current class as well.**

```python
class Cows(Mammals):
    def __init__(self, name: str):
        super().__init__(name=name, n_legs=4, sound="Moo")  # 👈 Call __init__ from Mammals

    def mooing(self):
        print(f"{self.sound} " * 5)

cow2 = Cows("Mali")
cow2.speak()
cow2.mooing()
```

In above code, you will see `super()`  keyword, `super()` let us to use parent class.

and now we finish all of about Python OOP. We will use all of this concept to apply with pytorch.