
'''
    @author Munawar Hasan <munawar.hasan@nist.gov>
'''

class GenericException(Exception):
    def __init__(self, msg):
        self.message = msg
        super().__init__(self.message)


class SquareMatrixException(Exception):
    def __init__(self, m, n):
        self.message = "matrix passed to model is not square: (" +str(m) +"," +str(n) +")"
        super().__init__(self.message)


class UnitaryMatrixQubitsException(Exception):
    def __init__(self, m, n, num_of_qubits):
        self.message = "num of qubits does not comply with matrix passed to model: (" +str(m) +"," +str(n) +")"
        super().__init__(self.message)


class InputException(Exception):
    def __init__(self):
        self.message = "input is none of empty"
        super().__init__(self.message)
