"""Experimental Parcels 4 RK4 kernel with longitude wrapping at every stage.

Not enabled in the production GLOENS workflow until validated.
Adapted from the documented Parcels 4 RK4 and NEMO periodic RK2 examples.
"""


def AdvectionRK4_periodicBC(particles, fieldset):
    """Fourth-order RK with periodic longitude at intermediate and final stages."""
    (u1, v1) = fieldset.UV[particles]
    x1 = ((particles.x + u1 * 0.5 * particles.dt + 180.0) % 360.0) - 180.0
    y1 = particles.y + v1 * 0.5 * particles.dt
    (u2, v2) = fieldset.UV[
        particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles
    ]
    x2 = ((particles.x + u2 * 0.5 * particles.dt + 180.0) % 360.0) - 180.0
    y2 = particles.y + v2 * 0.5 * particles.dt
    (u3, v3) = fieldset.UV[
        particles.t + 0.5 * particles.dt, particles.z, y2, x2, particles
    ]
    x3 = ((particles.x + u3 * particles.dt + 180.0) % 360.0) - 180.0
    y3 = particles.y + v3 * particles.dt
    (u4, v4) = fieldset.UV[
        particles.t + particles.dt, particles.z, y3, x3, particles
    ]
    dx = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * particles.dt
    dy = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * particles.dt
    particles.dx += ((particles.x + particles.dx + dx + 180.0) % 360.0) - 180.0 - (particles.x + particles.dx)
    particles.dy += dy
