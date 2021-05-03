import torch

# Other helper packages

from gpytorch import lazify
import matplotlib.pylab as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split



print(load_boston()["DESCR"])


# load datasets

X, y = load_boston(return_X_y=True)

# Train-test split. 
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=41 # Fixed for reproducibility
                                                   )

# Tensorise your data. Watch out for dimensionality issues.

train_x = torch.tensor(X_train).float()
train_y = torch.tensor(y_train).float()

test_x = torch.tensor(X_test).float()
test_y = torch.tensor(y_test).float()


class feature_map(torch.nn.Module):
    """
    Implement your own feature map function
    
    - Set up 2 hidden layers.
    """
    
    def __init__(self, input_dim, h_layer1, h_layer2, output_dim):

        super().__init__()
        # set up the feature map structure
        
    def forward(self, x):
        
        return None



class RBFkernel(torch.nn.Module):
    """
    An RBF kernel for illustration
    """
    
    def __init__(self):
        
        super().__init__()
        
        # Parameterise the lengthscale so you can learn it later
        
        self.lengthscale = torch.nn.Parameter(torch.tensor(1.).float())
        
    def forward(self, x1, x2=None):
        """
        if x2 is None, then k(x1) = k(x1, x1) returns the square kernel matrix
        otherwise k(x1, x2) returns a x1_dim \times x2_dim rectangular matrix
        
        """
        
        if x2 is None:
            x2 = x1
        
        return lazify(torch.exp(-torch.cdist(x1, x2, p=2)/self.lengthscale))



# Example with RBF Kernel

RBF = RBFkernel()
rbf = RBF(train_x)

# Compute log determinant

print("Log determinant:", rbf.logdet())
print("\n")

# Compute matrix inverse product (K+lambda I)^{-1} y

print(rbf.add_diag(torch.tensor(1e-3).float()).inv_matmul(train_y)[:5])


class DeepRBFkernel(torch.nn.Module):
    
    def __init__(self, input_dim, h_layer1, h_layer2, output_dim):
        
        super().__init__()
        
        # Fill in something here        
        
        
    def forward(self, x1, x2=None):
        
        # Fill in something here

        return None


class kernel_model(torch.nn.Module):
    
    def __init__(self, input_dim, h_layer1, h_layer2, output_dim):
        
        super().__init__()
        
        # Fill in something here

    def forward(self, x1):
        
        # Fill in something here
        
        return None
    

# Write up your own loss function:

def nll(km, y):
    """
    This is the negative loglikelihood score
    
    km must be a lazy tensor
    and y is a torch tensor
    """
    return None


# Set up the deep kernel object

input_dim = train_x.shape[1]
h_layer1, h_layer2 = [30, 50]
output_dim = 100

KM = kernel_model(input_dim, h_layer1, h_layer2, output_dim)

# Set up optimiser
optim = torch.optim.Adam(KM.parameters(), lr=1e-3)


# set up epoch
epoch = 1000

for i in range(epoch):

    # Zero gradients from previous iteration
    optim.zero_grad()
    
    # Output from model
    output = KM(train_x)
    
    # Calc loss and backprop gradients
    loss = nll(output, train_y.reshape(-1, 1))
    loss.backward()
    
    if i % 100 == 0:
        print('Iter get_ipython().run_line_magic("d/%d", " - Loss: %.3f' % (")
            i + 1, training_iter, loss.item()
        ))    
        
    optim.step()


pred = KM.deeprbf(test_x, train_x).evaluate() @ KM(train_x).inv_matmul(train_y.reshape(-1,1))


plt.plot(pred.detach().numpy(), test_y.numpy(), "x")
plt.plot(test_y, test_y, "-")
plt.xlabel("truth")
plt.ylabel("prediction")
plt.show()
