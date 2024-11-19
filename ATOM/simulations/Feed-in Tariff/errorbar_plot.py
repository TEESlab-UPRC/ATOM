""""
    Copyright (C) 2022 Technoeconomics of Energy Systems laboratory - University of Piraeus Research Center (TEESlab-UPRC)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
################################################################################
#                                                                              #
#                       Plotting Results using error bars                      #
#                                                                              #
################################################################################
#

def plot_errorbars(num_samples, capacity, string):
    # Processing the data - Statistics
    num = num_samples
    cap = capacity
    cap_mu = [0 for k in range(len(cap))]
    cap_var = np.zeros((2,len(cap)))
    #cap_var = [0 for k in range(len(cap)) for l in range(1)]
    for i in range(len(cap)):
        buff = [0 for k in range(num)]
        k = 0
        for j in range(num):
            buff[k] = cap[j][i]
            k += 1
            cap_var[0][i] = cap_mu[i] - np.min(buff)
        cap_mu[i] = np.median(buff)
        cap_var[1][i] = np.max(buff) - cap_mu[i]
        # Plotting the results
        t = [i for i in range(len(cap))]
        y = [400 for i in range(len(cap))]
        plt.errorbar(t, cap_mu, cap_var,  mfc='red', ms=20, mew=4, alpha = 0.3, animated = 0)
        plt.suptitle(string, fontsize=12)
        plt.xlabel('Month of simulation (2018-2025)', fontsize=10)
        plt.ylabel('PV capacity additions (MW)', fontsize=10)
        plt.plot(t, y, color='red', linestyle='dashed',linewidth=2, markersize=12)
        plt.show()
