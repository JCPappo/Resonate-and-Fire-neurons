import torch
import numpy as np

class RAF(torch.nn.Module):
    instances = []
    """List of instances created, used to initialize and clear neuron states when the
    argument `init_hidden=True`"""

    """
    Resonate and Fire neuron model.
    Input is assumed to be a current injection.
    Membrane potential decays exponentially with rate beta while oscillating with
        a given frequency


    :param frequency: Frequency of the damped oscillations. May be a single-valued tensor 
        (i.e., equal frequency for all neurons in a layer), or multi-valued
    :type frequency: float or torch.tensor

    :param beta: Membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight per
        neuron)
    :type beta: float or torch.tensor

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional

    :param threshold: Threshold for potential to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param V_reset: Reset value of the membrane's potential. Defaults to
        -threshold/2 in order to simulate a relative refractory period
    :type V_reset: float, optional

    :param spike_gradient: Surrogate gradient for the term dS/dU (differentiable 
        approximation to the true gradient). Defaults to ATan surrogate gradient. 
    :type spike_grad: surrogate gradient function, optional
    """

    def __init__(self, frequency, beta, init_hidden=None, output=None, threshold=None, V_reset=None, spike_gradient=None):
        super(RAF, self).__init__()

        RAF.instances.append(self)
        self.frequency = frequency
        self.beta = beta
        self.init_hidden = False if init_hidden is None else init_hidden
        self.output = False if output is None else output
        self.threshold = 1 if threshold is None else threshold
        self.V_reset = -self.threshold/2 if V_reset is None else V_reset
        self.spike_gradient = self.ATan.apply if spike_gradient is None else spike_gradient

        
        if self.init_hidden:
            self.I = self.init_RAF()
            self.V = self.init_RAF()
            self.spk = self.init_RAF()

    def init_RAF(self):
        """
        Used to initialize I, V and spk as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        param = _SpikeTensor(init_flag=False)

        return param

    def forward(self, input_, I=False, V=False):
        
        if hasattr(V, "init_flag"):  # only triggered on first-pass
            V = V._SpikeTorchConv(input_=input_)
            I = I._SpikeTorchConv(input_=input_)
        elif self.init_hidden and hasattr(
            self.V, "init_flag"
        ):  # init_hidden case
            self.V = self.V._SpikeTorchConv(input_=input_)
            self.I = self.I._SpikeTorchConv(input_=input_)
            self.spk = self.spk._SpikeTorchConv(input_=input_)

        if not self.init_hidden:
            I, V = self._calc_state(input_, I, V)
            I, V, spike = self.fire(I, V)

            return I, V, spike

        if self.init_hidden:
            self._raf_forward_cases(I, V)
            self.I, self.V = self._calc_state_hidden(input_)
            self.fire_hidden()
                           
            if self.output:  # read-out layer returns output+states
                return self.I, self.V, self.spk
            else:  # hidden layer e.g., in torch.nn.Sequential, only returns output
                return self.spk

    def fire(self, I, V):
        """Generates spike if V > threshold.
        Returns spk."""
        spike = self.spike_gradient((V-self.threshold)) #Read Atan class
        I, V = self._reset(I, V) 

        return I, V, spike

    def fire_hidden(self):
        """Used if `init_hidden=True`.
        Generates spike if self.V > threshold"""
        self.spk = self.spike_gradient((self.V-self.threshold))
        self._reset_hidden()
    
    def _calc_state(self, input_, I, V):
        """Updates the state of the system.
        Returns I and V."""
        I_ = I
        V_ = V
        I = self.beta*I_ - (1 - self.beta)*self.frequency*V_ + input_
        V = (1 - self.beta)*self.frequency*I_ + self.beta*V_

        return I, V

    def _calc_state_hidden(self, input_):
        """Used if `init_hidden=True`. 
        Updates the state of the system. 
        Returns I and V."""
        I_ = self.I
        V_ = self.V
        I = self.beta*I_ - (1 - self.beta)*self.frequency*V_ + input_
        V = (1 - self.beta)*self.frequency*I_ + self.beta*V_

        return I, V

    def _reset(self, I, V):
        """Neurons that fire reset I and V values.
        Returns I and V."""
        fire_mask = V >= self.threshold
        I[fire_mask] = 0
        V[fire_mask] = self.V_reset

        return I, V

    def _reset_hidden(self):
        """Used if `init_hidden=True`.
        Neurons that fire reset I and V values."""
        fire_mask = self.V >= self.threshold
        self.I[fire_mask] = 0
        self.V[fire_mask] = self.V_reset

    def _raf_forward_cases(self, I, V):
        if I is not False and V is not False:
            raise TypeError(
                "When `init_hidden=True`, RAF expects 1 input argument."
            )

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            cls.instances[layer].V.detach_()
            cls.instances[layer].I.detach_()
            cls.instances[layer].spk.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            cls.instances[layer].V = _SpikeTensor(init_flag=False)
            cls.instances[layer].I = _SpikeTensor(init_flag=False)
            cls.instances[layer].spk = _SpikeTensor(init_flag=False)

    class ATan(torch.autograd.Function):
      """Heaviside on the forward pass,
      surrogate gradient on the backward pass"""
      @staticmethod
      def forward(ctx, V):
          spike = (V > 0).float() 
          ctx.save_for_backward(V)  # store the membrane for use in the backward pass
          return spike

      @staticmethod
      def backward(ctx, grad_output):
          V, = ctx.saved_tensors  # retrieve the membrane potential
          grad = 1 / (1 + (np.pi * V)**2) * grad_output 
          return grad
    

class _SpikeTensor(torch.Tensor):
    """Inherits from torch.Tensor with additional attributes.
    ``init_flag`` is set at the time of initialization.
    When called in the forward function of any neuron, they are parsed and
    replaced with a torch.Tensor variable.
    """

    @staticmethod
    def __new__(cls, *args, init_flag=False, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *args,
        init_flag=True,
    ):
        # super().__init__() # optional
        self.init_flag = init_flag


    def _SpikeTorchConv(*args, input_):
        """Convert SpikeTensor to torch.Tensor of the same size as ``input_``."""
    
        states = []
        # if len(input_.size()) == 0:
        #     _batch_size = 1  # assume batch_size=1 if 1D input
        # else:
        #     _batch_size = input_.size(0)
        if (
            len(args) == 1 and type(args) is not tuple
        ):  # if only one hidden state, make it iterable
            args = (args,)
        for arg in args:
            arg = arg.to("cpu")
            arg = torch.Tensor(arg)  # wash away the SpikeTensor class
            arg = torch.zeros_like(input_, requires_grad=True)
            states.append(arg)
        if len(states) == 1:  # otherwise, list isn't unpacked
            return states[0]
    
        return states