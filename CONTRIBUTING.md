# Contributing to Legate-DataFrame

Contributions to `legate-dataframe` fall into the following categories:

1. To report a bug, request a new feature, or report a problem with documentation, please file an
   [issue](https://github.com/rapidsai/legate-dataframe/issues/new/choose) describing the problem or new feature
   in detail. The RAPIDS team evaluates and triages issues, and schedules them for a release. If you
   believe the issue needs priority attention, please comment on the issue to notify the team.
2. To propose and implement a new feature, please file a new feature request
   [issue](https://github.com/rapidsai/legate-dataframe/issues/new/choose). Describe the intended feature and
   discuss the design and implementation with the team and community. Once the team agrees that the
   plan looks good, go ahead and implement it.
3. To implement a feature or bug fix for an existing issue, please follow the [code
   contributions](#code-contributions) guide below. If you need more context on a particular issue,
   please ask in a comment.

As contributors and maintainers to this project, you are expected to abide by the code of
conduct. More information can be found at:
[Contributor Code of Conduct](https://docs.rapids.ai/resources/conduct/).
Code contributions must follow the guidelines below.

(code-contributions)=
## Code contributions

Please see the [README](README.md) for information on how to set up a development environment
and building legate-dataframe.

Information on the code layout and design can be found in the [developer guide](DEVELOPER_GUIDE.md).

### Testing

You can run tests on a local build using `./build.sh test`.  The CI setup will run tests on
a single GPU via `./ci/run_ctests.sh` and `./ci/run_pytests.sh`.
These launch legate with as many GPUs as your system has and `LEGATE_TEST=1`.

Because CI is limited to a single GPU run, always make sure to test distributed algorithms
on a machine with multiple GPUs.

### Pre-commit hooks

legate-dataframe uses [pre-commit](https://pre-commit.com/) to execute all code linters and formatters.
Any code contributions must pass the `pre-commit` checks and this is enforced by CI.

### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
