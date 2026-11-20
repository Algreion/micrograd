from math import tanh,exp,log,cos,sin
import random

class Value:
    def __init__(self, data: float, _children=(), _op = ''):
        """Node for a data structure that links math operations together (Base for NN)"""
        self.data = data
        self.grad = 0 # Derivative (gradient) affecting the final output | Base case (for output) is 1.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    def __repr__(self):
        return f"Value(data={self.data:.2f},grad={self.grad:.2f},op={self._op if self._op else None})"
    def __str__(self):
        return str(self.data)
    def __getitem__(self, index: int):
        return list(self._prev)[index]
    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward(): # Closure, applied during backpropagation
            self.grad += 1.0 * out.grad # dx of a+b = 1
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other): # other + self
        return self + other
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return (-self) + other
    def __abs__(self):
        return self if self.data >= 0 else -self
    def __le__(self, other):
        if not isinstance(other,Value): other = Value(other)
        return self.data <= other.data
    def __lt__(self, other):
        if not isinstance(other,Value): other = Value(other)
        return self.data < other.data
    def __ge__(self, other):
        return -self <= -other
    def __gt__(self, other):
        return -self < -other
    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad # dx of a*b = a
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other):
        return self*other
    def __pow__(self, number: float):
        if not isinstance(number, (int, float)): return NotImplemented
        out = Value(self.data**number, (self,), f'**{number}')
        def _backward():
            self.grad += number * self.data**(number-1) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        return self * other**(-1) # a/b = a * (1/b)
    def __rtruediv__(self, other):
        return self**(-1) * other
    def __iter__(self):
        return iter([self])

    def exp(self):
        x = self.data
        out = Value(exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad # dx of e^x = e^x
        out._backward = _backward
        return out
    
    def log(self):
        x = self.data
        out = Value(log(x), (self,), 'log')
        def _backward():
            self.grad += (1 / x) * out.grad # dx of ln(x) = 1/x
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = tanh(x)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad # dx of tanh(x) = 1-tanh(x)^2
        out._backward = _backward
        return out
    
    def cos(self):
        x = self.data
        out = Value(cos(x), (self,), 'cos')
        def _backward():
            self.grad += -sin(x) * out.grad
        out._backward = _backward
        return out
    
    def sin(self):
        x = self.data
        out = Value(sin(x), (self,), 'sin')
        def _backward():
            self.grad += cos(x) * out.grad
        out._backward = _backward
        return out
    
    def tan(self):
        x = self.data
        t = sin(x)/cos(x)
        out = Value(t, (self,), 'tan')
        def _backward():
            self.grad += (1 + t**2)*out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + exp(-x)) # sigmoid(x)
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1-s) * out.grad
        out._backward = _backward
        return out
    
    def backward(self) -> None:
        """Backpropagation with previous nodes"""
        def topological_sort(root):
            """Adds a node to the list only after all its children have been processed"""
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(root)
            return topo
        
        self.grad = 1.0 # Base case
        for node in reversed(topological_sort(self)):
            node._backward()

def explanation():
    pass
    #! Backpropagation
    # We have a final output L, and want to find how everything impacts it (derivatives)
    # Derivative of previous nodes (neurons!) is self-explanatory.
    # Going back one layer, we apply the chain rule, multiplying the derivatives together. | LOCAL GRADIENT!
    # ^ If dx=3 and dy=2, where x->y->L, then a tiny nudge to x causes 3 times the nudge in y, which in turn causes 2 times the nudge in L. Hence 3*2 = 6 for the impact x has to L.
    # This way, we can recursively go back and find how each neuron impacts the final output!
    # Example: x -> y -> L
    # dL/dy = 3, dy/dx = 2, then dL/dx = (dL/dy)*(dy/dx)=3*2 = 6 | You know how all nodes impact the final one.
    #* _backward() is a closure to apply local gradient onto previous nodes
    #* In this case, it is applied after a topological sort
    #* We must add the derivatives to the gradient since we have multiple variables.

    #* Nudging all values in the direction of their gradient, we increase the final output

class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> list:
        return []

class Neuron(Module):
    def __init__(self, nin: int):
        """Single NN Neuron. 
        nin = Number of expected inputs
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # Random weight for each input
        self.b = Value(random.uniform(-1,1)) # Starting bias for neuron | 0.01 if relu, random(-1,1) if tanh
    def __repr__(self):
        return f"Neuron(inputs={len(self.w)},weights=[{",".join(str(round(w.data,2)) for w in self.w)}],bias={self.b.data:.2f})"
    def __str__(self):
        return f"N({len(self.w)})"
    def __call__(self, input: list, activation: str = 'tanh') -> Value:
        """
        Get the neuron's activation given a group of inputs
        Activation options: tanh, sigmoid, relu, linear. Default is tanh.
        """
        # w * x + b
        output = sum((w_i*x_i for w_i,x_i in zip(self.w, input)), self.b)
        match activation.lower():
            case 'tanh'|'t':
                return output.tanh()
            case 'sigmoid'|'sig'|'s':
                return output.sigmoid()
            case 'relu'|'r':
                return output.relu()
            case 'linear'|'none'|'l':
                return output
            case _:
                raise ValueError('Invalid activation.')
    
    def parameters(self) -> list[Value]:
        """Returns all parameters from the neuron"""
        return self.w + [self.b]
    def _size(self) -> int:
        return len(self.parameters())

class Layer(Module):
    def __init__(self, nin: int, neurons: int):
        """Layer of NN Neurons.
        nin = Number of expected inputs.
        neurons = Number of neurons in the layer"""
        self.neurons = [Neuron(nin) for _ in range(neurons)]
    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]
    def __repr__(self):
        return f"Layer(inputs={len(self.neurons[0].w)},neurons={len(self.neurons)})"
    def __str__(self):
        return f"L({len(self.neurons)})"
    
    def __call__(self, input, activation: str = 'tanh') -> Value | list[Value]:
        """Returns the activations for all neurons in the layer."""
        out = [n(input, activation) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [p for n in self for p in n.parameters()]
    def _size(self) -> int:
        return len(self.parameters())

class MLP(Module):
    def __init__(self, nin: int, layers: list[int], activation: str = 'tanh', last_activation: str | None = None):
        """Core Neural Network.
        nin is number of inputs.
        layers is list for number of neurons per layer. Ex: [4, 4, 1]
        Activation options: tanh, sigmoid, relu, linear. Default is tanh.
        If last_activation isn't specified, it will be the same as the rest.
        """
        size = [nin] + layers
        self.layers = [Layer(size[i],size[i+1]) for i in range(len(layers))]
        self.activation = activation
        self.last_activation = activation if last_activation is None else last_activation
    def __call__(self, input: list) -> Value | list[Value]:
        """Activates the entire network on a given input."""
        if isinstance(input,(float,int,Value)): input = [input]
        if isinstance(input[0],(list,tuple)):
            return [self(l) for l in input]
        layers = len(self.layers)
        for i,layer in enumerate(self.layers):
            if i == layers-1: input = layer(input,self.last_activation)
            else: input = layer(input,self.activation) # Iteratively update the result for every layer
        return input # Result (for final layer, the output)
    
    def __len__(self) -> int:
        return self._size()
    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]
    def __repr__(self):
        return f"MLP(size={len(self.layers)},layers=[{", ".join(str(len(l.neurons)) for l in self.layers)}],activation={self.activation}{f',last_activation={self.last_activation}' if self.last_activation != self.activation else ''})"
    def __str__(self):
        res = ""
        res += f"[{" ".join("I" for _ in range(len(self[0][0].w)))}]\n"
        res += "\n".join(f"[{" ".join("N" for _ in l.neurons)}]" for l in self.layers[:-1])
        res += f"\n[{" ".join("O" for _ in self[-1].neurons)}]"
        return res
    
    def parameters(self) -> list[Value]:
        """Total weights and biases of the entire MLP"""
        return [p for l in self for p in l.parameters()]
    
    def save(self, file: str, validation: bool = True) -> None:
        """Save trained model on a file."""
        with open(file, 'w') as f:
            if validation: f.write(str(self).replace("\n","")+'\n')
            for p in self.parameters():
                f.write(str(p.data)+'\n')

    def load(self, file: str, validation: bool = True) -> None:
        """Load parameters from a file. Must be of the same type as original."""
        check = False
        if validation:
            with open(file, 'r') as f:
                first = f.readline().strip()
                if first.startswith("["):
                    check = True
                    if first != str(self).replace("\n",""):
                        raise ValueError("MLP doesn't match given model.")
        with open(file,'r') as f:
            if check: next(f)
            for p in self.parameters():
                p.data = float(f.readline())

    def _reset(self) -> None:
        """Resets all training, beware!"""
        for p in self.parameters():
            p.data = random.uniform(-1,1)
        self.zero_grad()
    def _size(self) -> int:
        """Returns the parameter count of the NN."""
        return len(self.parameters())

def mse(output: list, expected: list) -> Value:
    """Computes the MSE (cost) for a particular result."""
    lo,le = len(output),len(expected)
    if lo != le: raise ValueError(f'Length mismatch between output ({lo}) and expected ({le}).')
    if le==0: return Value(0)

    factor = le
    if isinstance(expected[0],(list,tuple)):
        tot = sum(mse(o,e) for o,e in zip(output,expected))
    else:
        tot = sum((v_e-v_o)**2 for v_o,v_e in zip(output,expected))
    return tot/factor

def one_hot(n: int, size: int = -1):
    """Return list representation of integers (0-indexed)."""
    if size == -1: size = n+1
    elif size < n+1: raise ValueError
    out = [0] * size
    out[n] = 1
    return out

def softmax(logits: list[Value]) -> list[Value]:
    """Computes softmax."""
    max_logit = max(v.data for v in logits) # Stability
    logits = [(v-max_logit).exp() for v in logits]
    norm = sum(logits)
    return [v/norm for v in logits]

def bce(p: Value | list[Value], expected: int | bool, sigmoid: bool = False) -> Value:
    """Returns binary cross-entropy loss given a probability p. Expected should be 0 or 1.
    If sigmoid is True, p will be treated as a raw logit."""
    if isinstance(p,list):
        lp,le = len(p),len(expected)
        if lp!=le: raise ValueError(f'Length mismatch between p ({lp}) and expected ({le}).')
        return sum(bce(v,e,sigmoid) for v,e in zip(p,expected))/lp
    if sigmoid: p = p.sigmoid()
    return -p.log() if expected else -(1-p).log()

def cce(logits: list[Value | list], expected: int | list[int]) -> Value:
    """Returns categorical cross-entropy loss. Expected should be an index or one-hot vector."""
    if len(logits)>0 and isinstance(logits[0],list):
        ll,le = len(logits),len(expected)
        if ll!=le: raise ValueError(f'Length mismatch between logits ({ll}) and expected ({le}).')
        return sum(cce(l,e) for l,e in zip(logits,expected))/ll
    if isinstance(expected,list): expected = expected.index(1)
    return -softmax(logits)[expected].log()

def train(network, inputs: list[list | int | float], outputs: list[list | int | float], step: float = 0.1, cost: str = 'mse') -> list[Value]:
    """Single step of NN training on a input-output batch.
    Cost: mse, bce, cce. Default is mse.
    """
    if not isinstance(inputs,list): inputs = [inputs]
    if not isinstance(outputs,list): outputs = [outputs]
    match cost.lower():
        case 'bce': f = bce
        case 'cce': f = cce
        case 'mse': f = mse
        case _: raise ValueError('Invalid cost function. Choose mse, bce or cce.')
    result = [network(i) for i in inputs]
    c = f(result, outputs)
    network.zero_grad()
    c.backward()
    for p in network.parameters():
        p.data -= step * p.grad
    return result

def train_xor(multi_output: bool = False, iterations: int = 1000, step: float = 0.5, info: bool = True,
    hidden_layers: list | None = None, activation: str = 'tanh', last_activation: str = 'sigmoid') -> MLP:
    """
    Mock NN training to compute XOR.
    If multi_output, computes all possible XOR combinations: (x0,x1), (x0,x2), (x1,x2).
    """
    if hidden_layers is None: hidden_layers = [4,4]
    def xor(a: int, b: int) -> int:
        return int((a and not b) or (not a and b))
    def check(outputs: list, expected: list) -> tuple:
        if multi_output: # Flattening
            expected = [e for ex in expected for e in ex]
            outputs = [o for out in outputs for o in out]
        total = len(expected)
        correct = sum(round(o.data)==e for o,e in zip(outputs,expected))
        return (correct,total,correct/total)
    
    if multi_output:
        layers = hidden_layers+[3]
        X = [[a,b,c] for a in range(2) for b in range(2) for c in range(2)]
        Y = [[xor(x[0],x[1]), xor(x[0],x[2]), xor(x[1],x[2])] for x in X]
        nin = 3
    else:
        layers = hidden_layers+[1]
        X = [[a,b] for a in range(2) for b in range(2)]
        Y = [xor(*x) for x in X]
        nin = 2
    nn = MLP(nin,layers,activation=activation,last_activation=last_activation)
    
    if info: start = check(nn(X),Y)

    for k in range(iterations):
        # Forward pass
        result = nn(X)
        loss = bce(result, Y)

        # Backward pass
        nn.zero_grad()
        loss.backward()

        # Update
        for p in nn.parameters():
            p.data -= step * p.grad
        if info and (k==0 or not (k+1)%100): print(f"{k+1}. Loss: {loss.data:.3f}")
    if info:
        print()
        sc,st,sp = start
        print(f"Initialization: {sp*100:.1f}% ({sc} out of {st} correct)")
        ec,et,ep = check(nn(X),Y)
        print(f"Post-training: {ep*100:.1f}% ({ec} out of {et} correct)")
    return nn