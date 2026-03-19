from dataclasses import dataclass

@dataclass
class PendulumOnCartConfig:
    """
    Configuration class for the inverted pendulum on cart system.

    Parameters
    ----------
    m_cart : float, optional
        Mass of the cart, by default 1.0
    m_pole : float, optional
        Mass of the pendulum, by default 0.1
    length : float, optional
        Length of the pendulum, by default 0.8
    gravity : float, optional
        Gravitational acceleration, by default 9.81
    damping : float, optional
        Damping coefficient, by default 1.0
    """
    m_cart: float = 1.0
    m_pole: float = 0.1
    length: float = 0.8
    gravity: float = 9.81
    damping: float = 1.0