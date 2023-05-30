ELQR Examples
=============

.. contents:: Examples
   :depth: 2
   :local:


Gaussian Mixture Based
----------------------
The multi-agent ELQR algorithm can be run for a finite horizon with non-linear
dynamics and a use a Gaussian Mixture agent and target density with the following.
This leverages the single agent ELQR formulation from GNCPy.

.. literalinclude:: /examples/guidance/elqr.py
   :linenos:
   :pyobject: modify_quadratize

which gives this as output.

.. image:: /examples/guidance/elqr_modify_quadratize.gif
   :align: center


Optimal Sub-Pattern Assignment Based
------------------------------------
The multi-agent ELQR algorithm can be run for a finite horizon with non-linear
dynamics and a use an OSPA based non-quadratic cost with the following.
This leverages the single agent ELQR formulation from GNCPy.

.. literalinclude:: /examples/guidance/elqr.py
   :linenos:
   :pyobject: elqr_ospa

which gives this as output.

.. image:: /examples/guidance/elqr_ospa_cost.gif
   :align: center


OSPA Based with Obstacles
-------------------------
The multi-agent ELQR algorithm can be run for a finite horizon with non-linear
dynamics and a use an OSPA based non-quadratic cost plus an additional cost
to avoid obstacles with the following. This leverages the single agent ELQR
formulation from GNCPy.

.. literalinclude:: /examples/guidance/elqr.py
   :linenos:
   :pyobject: elqr_ospa_obstacles

which gives this as output.

.. image:: /examples/guidance/elqr_ospa_cost_obstacles.gif
   :align: center
