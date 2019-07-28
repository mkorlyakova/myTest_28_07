# Test code


Introduction: In mathematics, a conic section is a curve obtained as the intersection of the surface of a cone with a plane. The three types of conic section are the hyperbola, the parabola and the ellipse. Algebraically it can be written as Ax2+Bxy+Cy2+Dx+Ey+F=0, where not all of A, B and C are zero. If the conic section is non-degenerate, it can be classified in terms of the value B2-4ACcalled discriminant:
Ellipse if B2-4AC<0.
Parabola if B2-4AC=0.
Hyperbola if B2-4AC>0.

Subject: Using the methods of machine learning, develop an application that can determine the type of conic section according to the specified coefficients. Generate the training and testing datasets yourself.

Limitations: The application should work correctly for input coefficients not exceeding 1 in absolute value.

Bonus: Develop functional auto-tests.

feature_generate.py - generate data for my test and train (train.csv, test.csv)
feature_and_model_select.py - select my solver model (my experiment with this dataset)
clalassifier_train_and_test.py - final solver generate and write testres.txt

test.csv = test data
train.csv - training data




