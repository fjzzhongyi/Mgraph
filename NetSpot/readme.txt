NetSpot 1.0.1
------------------------

This tools finds anomalous regions in dynamic networks. The implemented 
algorithm is described in the following paper:

Misael Mongiovi, Petko Bogdanov, Razvan Ranca, Ambuj K. Singh, Evangelos E. Papalexakis, Christos Faloutsos. NetSpot: Spotting Significant Anomalous Regions on Dynamic Networks. SIAM International Conference on Data Mining. 2013. 

You may also be interested in the following related papers:

Misael Mongiovi, Petko Bogdanov, Ambuj K. Singh. Mining evolving network processes. IEEE International Conference on Data Mining. 2013.

Petko Bogdanov, Misael Mongiovi, Ambuj K. Singh. Mining Heavy Subgraphs in Time-Evolving Networks. IEEE International Conference on Data Mining. 2011.

If you use this software please consider citing the above papers.

[License Agreement]
Please read this agreement carefully before using this software and its
accompanied dataset. By using the software and the dataset enclosed in 
this package (NetSpot), you agree to the terms of this license.
If you do not agree to the terms of this license, please delete this
package and DO NOT USE it.
(1) This copy is for your internal use only. Please DO NOT redistribute it. 
The software is available at http://www.cs.ucsb.edu/~dbl/software.php. 
Please DO NOT post this package on the internet. 
(2) As unestablished research software, this software is provided on an 
``as is'' basis without warranty of any kind, either expressed or 
implied. The downloading, or executing any part of this software 
constitutes an implicit agreement to these terms. These terms and 
conditions are subject to change at any time without prior notice.
No any other usage of the package is allowed without a written permission 
from the authors. It can NOT be used for any commercial interest.
(3) The authors do not hold any responsibility for the correctness of the 
software and datasets.
(4) The authors appreciate it if you can send us your feedback and test results.

3. COPYRIGHT
ALL copyrights of the software are reserved by authors.


CONTAINS:
------------------------

- SigSpot.jar: executable
- wikipedia990.quadruples: example of input file

USAGE:
------------------------

> java -jar SigSpot.jar wikipedia990.quadruples 10 10 out.txt

For a complete list of parameters run:

> java -jar SigSpot.jar

------------------------
Misael Mongiovì
mongiovi@dmi.unict.it
