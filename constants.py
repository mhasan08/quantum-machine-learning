
'''
    @author Munawar Hasan <munawar.hasan@nist.gov>
'''


class ExceptionMessages:
    ERR_QUBITS = "num-of-qubits is none"
    ERR_PRED_INDEX = "prediction index is not a dictionary"


class WarningMessages:
    WR = "Warning: "
    WR_UNITARY = "unitary matrix is none; generating a random unitary matrix ...."
    WR_PRED_INDEX = "prediction index is none, generating prediction index ...."


class Messages:
    METRICS = "calculating metrics ....."
    PRED_INDEX = "using prediction index as "
    PREDICTION_IS_SET_FALSE = "pred is false, will return quantum state probabilities and index"
    DELIMITER = "\t"


class QISKIT:
    SIMULATOR = "qasm_simulator"


class Environment:
    OCT = "octave"
    PATH = "PATH"
