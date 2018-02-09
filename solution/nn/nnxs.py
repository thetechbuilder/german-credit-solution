#Licence: Public Domain
#
#nnxs.py
#
"""Implements a neural network XML-serializer using xml.etree.ElementTree

Unit tests are in the current directory (test_nnxs.py)"""
#standard modules:
import xml.etree.ElementTree as ET
#BufferedIOBase is a base class for binary streams that support some 
#kind of buffering. It inherits IOBase:
from io import BufferedIOBase as _BufferedIOBase
#included modules:
from neuro import Network, NeuronView, Algorithms

__all__ = ["NNXSerializer"]

class NNXSerializer:
    """
    Provides a trivial API for serialization to and from standard XML
    It helps to parse/write your neural network (neuro.Network) object 
    from/to a specified file
    """
    #attributes:
    TAG_ROOT = "net"
    TAG_LAYER = "layer"
    TAG_NEURON = "neuron"
    ATTR_SPEED = "speed"
    ATTR_BIAS = "bias"
    ATTR_WEIGHTS = "weights"
    ATTR_ALG = "algorithm"

    #functions:
    @staticmethod
    def _validate_file_argument(target):
        """
        Internal method that has been designed to eliminate repeated code
        """
        if not isinstance(target, str) and not isinstance(
                target, _BufferedIOBase):
            raise TypeError("The type of the target value is wrong. "
                    "'target' must be either a filename string or a "
                    "binary file object, not {}".format(type(target)))

    @staticmethod
    def write(source, target):
        """
        Writes the specified network to file

        * "source" is an instance of "Network" or of subclass thereof
        * "target" is ether a filename or a binary file object
        """
        if not isinstance(source, Network):
            raise TypeError("The neural network to serialize must be "
                    "an instance of Network, or of a subclass thereof, "
                    "Not %s" % type(source))
        NNXSerializer._validate_file_argument(target)
        #root atributes countains overall parameters and layers
        root = ET.Element(NNXSerializer.TAG_ROOT, attrib = {
                    NNXSerializer.ATTR_SPEED : str(source.speed),
                    NNXSerializer.ATTR_ALG : source.algorithm.__name__})
        #SubElement function provides a way to create new sub-elements for
        #a given element.
        for layer in source._links: #traverse all layers
            sub = ET.SubElement(root, NNXSerializer.TAG_LAYER)
            for neuron in map(NeuronView, layer):
                ET.SubElement(sub, NNXSerializer.TAG_NEURON, 
                        attrib = {
                            NNXSerializer.ATTR_WEIGHTS: ", ".join(
                                map(str, neuron.weights)), 
                            NNXSerializer.ATTR_BIAS: str(neuron.bias)})
        #When encoding is US-ASCII or UTF-8 ET's output is binary!
        #Because the output is binary only BufferedIOBase-like objects 
        #are accepted.
        ET.ElementTree(root).write(target)

    @staticmethod
    def parse(target):
        """
        parse(target) -> Network object

        Loads an external XML-preserved neural network into a "Network" 
        object
        * "target" is ether a filename or a file object
        """
        NNXSerializer._validate_file_argument(target)
        root = ET.parse(target).getroot()
        L = [
                [
                    NeuronView.merge(
                        map(float, neuron.get(
                            NNXSerializer.ATTR_WEIGHTS).split(",")), 
                        float(neuron.get(NNXSerializer.ATTR_BIAS))) 
                    for neuron in layer]
                for layer in root]
        speed = float(root.get(NNXSerializer.ATTR_SPEED))
        alg = root.get(NNXSerializer.ATTR_ALG)
        if hasattr(Algorithms, alg):
            return Network(*L, n = speed, 
                    algorithm = getattr(Algorithms, alg)) 
        return Network(*L, n = speed)
