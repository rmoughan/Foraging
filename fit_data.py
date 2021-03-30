import pandas as pd
from model import ThompsonSampler as TS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# load in data
df = pd.read_pickle("data/treatment_A.pkl", compression="infer")
df.loc[df.flow_group == 0, "rewarded"] = 0

# set learning rates
t = np.arange(0, 150)
learning_rates = np.array([[1, 1], [0.5, 0.1], [0.10, 0.00]])
J = len(learning_rates)

with PdfPages('TSModel.pdf') as pdf:
    for bee in df.id.unique():
        # pull phase 1 data
        x = df.loc[df.id == bee, "flow_group"].values
        r = df.loc[df.id == bee, "rewarded"].values
        # need to subtract 1 so the arm IDs are [0, 1, 2]
        x_phase1 = x[:150] - 1
        r_phase1 = r[:150]
        fig, ax = plt.subplots(1, J + 1, sharey=True, sharex=True,
                               figsize=(5*(J+1), 5), tight_layout=True)

        # for each set of learning rates, fit a new model
        for j in range(J):
            model = TS(x=x_phase1,
                       r=r_phase1,
                       a0=[1, 1, 1],
                       b0=[1, 1, 1],
                       n_samples=5000,
                       lr=learning_rates[j])
            model.fit()
            post = np.array(model.posterior_history)
            for i in range(3):
                label = "p(arm %i| a, b)" % (i + 1)
                ax[j].plot(t, post[:, i], label=label)

            title = "a_lr = %.2f, b_lr = %.2f" % (learning_rates[j, 0], learning_rates[j, 1])
            ax[j].set_title(title)
            ax[j].legend()

        # plot the real data
        for k in range(3):
            label = "p(arm %i| a, b)" % (k + 1)
            real_pulls = np.cumsum(x_phase1 == k)/(t+1)
            ax[J].plot(t, real_pulls, label=label)
            ax[J].legend()
        ax[J].set_title("Real Data")
        fig.suptitle(bee)
        pdf.savefig(fig)
        plt.close()
