# HCIM
Hybrid Coastal Inundation Model
Developed for my MSc thesis submitted in fulfilment of the requirements for the MSc and the Diploma of Imperial College London.


                                                                  Abstract
                                                                  
                                                                 
A large part of the world population resides in coastal regions. The rise in sea levels due to global warming and the increase in the frequency of extreme events in coastal
areas poses a serious threat to them in the form of flooding. This warrants the availability of a fast and accurate computational solver to predict coastal flooding. The
complex coastal processes make it difficult to model the phenomenon numerically. Further, numerical models often have a trade-off between speed and accuracy. 

 In this study, a hybrid computational model based on shallow water equations is proposed. The computational domain is divided into three zones based on significant physical processes. In the first zone, deep water waves start propagating into
the domain and one layer non-hydrostatic shallow water equations are implemented. As the waves propagate onto the beach slope they start to steepen and break, marking the first division of the computational domain. In the second zone, hydrostatic
shallow water equations are implemented which allows the wave to transform into a bore-like shape to simulate the effects of wave breaking and compensate for the
low vertical resolution. The next division is made where the waves over-top the sea defence and flood the coastal region. Here, the flow behaviour is in the form
of non-dispersive fluxes and a hydrostatic shallow water model without advection is implemented. Further, to improve the accuracy of the results, a set-up correction is
implemented.

 The proposed numerical model is developed in Python3 in a modular fashion. The language is selected because it is modern and allows for easy development and coupling with other models in the future. The model is validated for various cases with
an existing robust model SWASH and is found to produce accurate results. 

 Further work to be conducted on this involves the implementation of parallel processing and methods to resolve irregular bathymetries. The fully developed model
will be able to perform real-time flood estimates, nearshore modelling, and flood risk analysis in national scales.
