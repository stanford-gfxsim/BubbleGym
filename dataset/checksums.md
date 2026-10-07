# Checksums

MD5 of the text of every tracked file. The repository ships LF, so the **LF**
column is what you should get; the **CRLF** column is the same bytes with every
newline rewritten, which is what an editor or a checkout that converts endings
leaves behind, so a converted copy can still be identified rather than looking
corrupt. For a compressed scene these are the checksums of the *decompressed*
text:

```bash
xz -dc dataset/exhalation/trackedBubInfo_NN.txt.xz | md5sum   # LF column
md5sum dataset/bubble_theater/curl_noise/trackedBubInfo_NN.txt
```

| file | lines | bytes (LF) | MD5 (LF) | MD5 (CRLF) |
| --- | ---: | ---: | --- | --- |
| `bubble_theater/curl_noise/trackedBubInfo_BEM.txt` | 243 | 7,712 | `95bda8a1ddc597c9a04f6220e85e3301` | `44a11f47dca10f800d1fa3754671ad1e` |
| `bubble_theater/curl_noise/trackedBubInfo_Ellipsoid.txt` | 243 | 7,712 | `a535df59dcf0948a7853d5e8097bef24` | `aae6b018184a865155a30a27cf4caa61` |
| `bubble_theater/curl_noise/trackedBubInfo_Minnaert.txt` | 243 | 7,712 | `20df72dd3b3fa9edc2f2673ae79ca63c` | `3b131fea3bb26c1a2c39bb8a90837c0c` |
| `bubble_theater/curl_noise/trackedBubInfo_NN.txt` | 243 | 7,712 | `2a240c405d27927c77e27b4f727d72f5` | `b6d437a5d8abe7f9835f10c4773d74e5` |
| `bubble_theater/ellipsoid/trackedBubInfo_BEM.txt` | 243 | 7,712 | `8e79fe9594d80a143b060e86c67e3dfa` | `15d4b4f81893ab6ee6d68e7a63729dab` |
| `bubble_theater/ellipsoid/trackedBubInfo_Ellipsoid.txt` | 243 | 7,712 | `fe579f0dddbd9a530be61eb38caa114e` | `860198ebb52c6bf7610c2c68f4742426` |
| `bubble_theater/ellipsoid/trackedBubInfo_Minnaert.txt` | 243 | 7,712 | `20df72dd3b3fa9edc2f2673ae79ca63c` | `3b131fea3bb26c1a2c39bb8a90837c0c` |
| `bubble_theater/ellipsoid/trackedBubInfo_NN.txt` | 243 | 7,712 | `eaf370aba2a55691233df17e6e8b82a6` | `7380fedc66b07b77267ea97d4a1c1194` |
| `bubble_theater/enright_test/trackedBubInfo_BEM.txt` | 123 | 3,882 | `4d035376da43b74fb4cf0b094e9fe65d` | `dd1b6f7578e96621528a8975c8e32bad` |
| `bubble_theater/enright_test/trackedBubInfo_Ellipsoid.txt` | 123 | 3,882 | `92f34db71376b467188854e9fa6c292a` | `05c5226c5830f8f002d122dd4b38a580` |
| `bubble_theater/enright_test/trackedBubInfo_Minnaert.txt` | 123 | 3,882 | `d85cf28f5de6b7bfff87ad573b57c7f8` | `153981bc2c74eda0ab8fe6e8769876b6` |
| `bubble_theater/enright_test/trackedBubInfo_NN.txt` | 123 | 3,882 | `23842a7d1c51f79483fa7cabbe76f6ea` | `43b547cc90d5fbfcf994c5c6a00be100` |
| `single_rising_bubble/trackedBubInfo_BEM.txt.xz` | 201,298 | 12,352,175 | `12033dc97fa51fb764c1899de4ef61cf` | `e2c09f9afdec3bc3d3ed5a75e68198ed` |
| `single_rising_bubble/trackedBubInfo_Minnaert.txt.xz` | 201,298 | 11,770,134 | `f2a9ebccbb4ce9bc5daa60977ca3ef39` | `80057cea83851d08f4189b9e8ed0e6d6` |
| `single_rising_bubble/trackedBubInfo_NN.txt.xz` | 201,298 | 12,351,993 | `b73c513e0ccd7b8f0452cc77d7b1e214` | `352bc59a14d4c18ca2fced963b86d77a` |
| `fruit_splash/trackedBubInfo_BEM.txt.xz` | 6,359,070 | 408,718,781 | `c76b6c7b150cf2a9be9025ee072a1cce` | `1a6276a01169b0ae8655e4346524d6f9` |
| `fruit_splash/trackedBubInfo_Minnaert.txt.xz` | 6,359,070 | 388,551,793 | `9ae3097deb876b9055c6604889a905a6` | `8fa9589cb8d319a71a3aa8d16efe6fd2` |
| `fruit_splash/trackedBubInfo_NN.txt.xz` | 6,359,070 | 408,642,295 | `259f89fa103a13a9d57eaed1081753be` | `a8c557ab34b62c10966522ee4e8a6602` |
| `exhalation/trackedBubInfo_BEM.txt.xz` | 21,222,779 | 1,321,234,256 | `dfd65e7748aa9f405182e92f355cfd4c` | `23f47e7e0474ed186472f6bbb15c7d29` |
| `exhalation/trackedBubInfo_Minnaert.txt.xz` | 21,222,779 | 1,253,228,293 | `57455e6447ff955a7266716352de095c` | `4fb22684a7f1471f2459be48765aaae7` |
| `exhalation/trackedBubInfo_NN.txt.xz` | 21,222,779 | 1,321,310,331 | `4a79ec6f3e11c8a18fcd964120569763` | `d5ac121c40fb638cb4354f0e87bd15f2` |
