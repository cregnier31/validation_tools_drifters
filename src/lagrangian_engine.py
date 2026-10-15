from parcels_tools import run_parcels_Drift
from opendrift_tools import run_opendrift

def run_lagrangian(engine, *args, **kwargs):
    """
    Run the selected Lagrangian engine.
    """

    engine = engine.lower()

    if engine == "parcels":
        return run_parcels_Drift(
            *args,
            **kwargs
        )

    elif engine == "opendrift":
        return run_opendrift(
            *args,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown Lagrangian engine: {engine}"
        )
