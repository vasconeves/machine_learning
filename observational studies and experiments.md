# Mammography study

Here I will demonstrate how we can analyze data from observational studies using a mammography study as an example.

The key questions here is the following: *does mammography speeds up cancer detection?*

*How do we set up an experiment in order to minimize the problem of confounding?*

This study consists in the screening of women's breasts by X-rays as shown in the Table below.

![Alt text](mammography_table.png)

We are inclined to compare between those who took the treatment with the ones how refused it. However this is an **observational** comparison!

Instead we need to compare the whole treatment group against the whole control group.

We need to do an **intent-to-treat** analysis.

From the table we will assume that the **outcome variable** will be death by breast cancer. This variable will depend on the treatment variable, that will be the people offered mammography.

Why? Because we cannot **force** people to do the mammography!

## RCT

In this experimental design we need to consider the following properties:

* Patient selection: Some populations are more likely to develop breast cancer than others depending on their prior health background, so the interpretation of any result we obtain will depend on how we have defined our selection procedure. In general, how we select the treatment and control groups will influence the population for which the conclusion is valid.

* Control group: We need to compare the outcome variable for those who have received the treatment with a baseline (i.e, the control group). Here, the control group (who were not offered a mammography) must be a comparable set of people to the treatment group (who have been offered a mammography).

* Features of the patients: One way to make accurate comparison and interpretation of results is to ensure that the treatment group is representative across factors such as health status, age, and ethnicity, and is similar along these dimensions as the control group. This way, we can attribute any differences in outcome to the treatment rather than differences in covariates. In a general study, it is upon the researchers' discretion to determine which features to be careful about.

An experimental design where treatments are assigned at random and that satisfies the three points described above is called a **randomized controlled trial (RCT)**. The controls can be an observational group or treated with a placebo.

## Double blind

In any experiment that involves human subjects, factors related to human behavior may influence the outcome, obscuring treatment effects. For example, if patients in a drug trial are made aware that they actually received the new treatment pill, their behavior may change in a number of ways, such as by being more or less careful with their health-related choices. Such changes are very difficult to model, so we seek to minimize their effect as much as possible.

The standard way to resolve this is through a double-blind study , also called a blinded experiment . Here, human subjects are prevented from knowing whether they are in the treatment or control groups. At the same time, whoever is in charge of the experiment and anyone else who could interact with the patient are also prevented from directly knowing whether a patient is in the treatment or the control group. This is to prevent a variety of cognitive biases such as observer bias or confirmation bias that could influence the experiment.

In some cases, it is impossible to ensure that a study is completely double-blind. In the mammography study, for example, patients will definitely know whether they received a mammography. If we modify the treatment instead to whether a patient is offered mammography (which they could decline), then we neither have nor want double-blindness.

## Hypothesis testing

From the table we can observe that

* death rate from breast cancer in control group = $\frac{63}{31k}$
* death rate from breast cancer in treatment group = $\frac{39}/{31k}$

Key question: *Is the difference in death rates between treatment and control sufficient to establish that mammography reduces the risk of death from breast cancer?*

We need to perform an **hypothesis test**.

Hypothesis testing steps:






