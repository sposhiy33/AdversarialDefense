import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight") 

acc = [0.6847, 0.6952, 0.6997, 0.7022, 0.7052, 0.7061, 0.7059, 0.705, 0.7029]
sig = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6] 
sig_zoom = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 ]

plt.plot(sig_zoom, acc)
plt.title("accuracy vs sigma (0.2 ep)")
plt.xlabel('sigma value')
plt.ylabel('accuracy')
plt.show()