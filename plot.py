import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# centralized training
c_map = [0.054,0.5431,0.5516,0.5363,0.5828,0.6481,0.6369,0.7298,0.6381,0.7582,0.7847,0.8007,0.8192,0.8109,
         0.8425,0.8299,0.8467,0.8598,0.8195,0.8666,0.8426,0.8399,0.8868,0.8608]
c_x = [i for i in range(len(c_map))]
# print(c_x)
# plt.plot(c_x, c_map)
# plt.xlabel(xlabel="Epoch")
# plt.ylabel(ylabel="Test mAP")
# plt.savefig("./plt_results/centralized_map")
# plt.show()

# FL training
fl_map = [0,0.15358397,0.641382055,0.736635758,0.734843871,0.788049031,
          0.827798031,0.833866401,0.839547463,0.8468214453382131,0.8481874350005869,0.8682464583704723,0.8636252359811284,
          0.854513807077555,0.8650362987688013,0.8607439816772184]
# fl_x = [i for i in range(len(fl_map))]
# print(fl_x)
# plt.plot(fl_x, fl_map)
# plt.xlabel(xlabel="Rounds")
# plt.ylabel(ylabel="Test mAP")
# plt.savefig("./plt_results/fl_map")

# client1 training
client_map = [0.04501, 0.123, 0.3789, 0.3679,0.3964,0.4996,0.5757,0.6478,0.6216,0.661,0.7145,0.7468,0.742,0.709,
              0.7229,0.7717,0.7655,0.79,0.7739,0.781]
c1_x = [i for i in range(len(client_map))]
print(c1_x)
plt.plot(c1_x, client_map)
x_major_locator=MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlabel(xlabel="Epoch")
plt.ylabel(ylabel="Test mAP")
plt.savefig("./plt_results/c1_map")