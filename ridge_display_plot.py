def display_plot(cv_scores, cv_scores_std):
    """
    Plot cross-validated R2 for range of different alphas in a Ridge linear model.

    :param cv_scores: array containing the resulting cross-validated R2.
    :param cv_scores_std: cv_scores standard deviation.

    :return: plots R2 and standard error for each alpha.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


