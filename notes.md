
We have a final output L, and want to find how everything impacts it (derivatives).
Derivative of previous nodes (neurons) is self-explanatory.
Going back one layer, we apply the chain rule, multiplying the derivatives together. Local gradient.
^ If dy/dx=3 and dL/dy=2, where x->y->L, then a tiny nudge to x causes 3 times the nudge in y, which in turn causes 2 times the nudge in L. Hence 3*2 = 6 for the impact x has to L.
This way, we can recursively go back and find how each neuron impacts the final output!
Example: x -> y -> L
dL/dy = 2, dy/dx = 3, then dL/dx = (dL/dy)*(dy/dx)=2*3 = 6 | You know how all nodes impact the final one.
_backward() is a closure to apply local gradient onto previous nodes.
In this case, it is applied in reverse topological order.
We must add the derivatives to the gradient since contributions of all nodes must be summed.
Nudging all values in the direction of their gradient, we increase the final output.