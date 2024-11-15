import matplotlib.pyplot as plt

x = [100, 500, 1000, 5000, 10000]
y1 = [74.00, 76.40, 79.10, 80.00, 80.08]
y2 = [0.634148558974266, 0.557537654042244, 0.5104934811592102, 0.4837458670139313, 0.47866792380809786]

plt.semilogx(x, y1)
plt.axhline(y=80, color='r', linestyle='--')
plt.xlabel('Samples')
plt.ylabel('Accuracy')
plt.title('Accuracy of Training Samples')
plt.grid()

plt.show()

plt.semilogx(x, y2)
plt.xlabel('Samples')
plt.ylabel('Loss')
plt.title('Loss of Training Samples')
plt.grid()

plt.show()
