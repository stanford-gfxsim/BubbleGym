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
| `bubble_theater/curl_noise/trackedBubInfo_BEM.txt` | 183 | 5,815 | `abd550bbacfcca076b6d703075577a97` | `b75d50b48f5f1eb5882f56fa9fd99b11` |
| `bubble_theater/curl_noise/trackedBubInfo_Ellipsoid.txt` | 183 | 5,815 | `750f212e4ecc14471354429f5920c1e1` | `218ac8c1d6248ff1d4eea7194bdf8aef` |
| `bubble_theater/curl_noise/trackedBubInfo_Minnaert.txt` | 183 | 5,815 | `3eb1241d631809f632399060693f0c27` | `b4c61958b59707a17eb7cece859809f9` |
| `bubble_theater/curl_noise/trackedBubInfo_NN.txt` | 183 | 5,815 | `3879dcbffe9d4a81a2d04f49ee681a32` | `789ed049ed4737530968b63bd06643b0` |
| `bubble_theater/ellipsoid/trackedBubInfo_BEM.txt` | 183 | 5,815 | `2e0efd995537b015455201b6c7a4b01f` | `92a393301de7258611b3b9eaf89100c8` |
| `bubble_theater/ellipsoid/trackedBubInfo_Ellipsoid.txt` | 183 | 5,815 | `fbca8a1a9def55c59c4cea0eafcd2ebb` | `b7c4643d582443ca939a7dcecfb91f17` |
| `bubble_theater/ellipsoid/trackedBubInfo_Minnaert.txt` | 183 | 5,815 | `3eb1241d631809f632399060693f0c27` | `b4c61958b59707a17eb7cece859809f9` |
| `bubble_theater/ellipsoid/trackedBubInfo_NN.txt` | 183 | 5,815 | `f1827a34ab0de682497c136b8dc2545b` | `e74ae6bc2e39562fadfc47d0156f984e` |
| `bubble_theater/enright_test/trackedBubInfo_BEM.txt` | 123 | 3,882 | `71cdc2e0776723f4c1c2504229927fef` | `1405ef052139e2223d2b615572cda479` |
| `bubble_theater/enright_test/trackedBubInfo_Ellipsoid.txt` | 123 | 3,882 | `8a70dded8fe58867ce2422dc15177c3e` | `206b2764d8247360d39acef1ac99d930` |
| `bubble_theater/enright_test/trackedBubInfo_Minnaert.txt` | 123 | 3,882 | `d85cf28f5de6b7bfff87ad573b57c7f8` | `153981bc2c74eda0ab8fe6e8769876b6` |
| `bubble_theater/enright_test/trackedBubInfo_NN.txt` | 123 | 3,882 | `a99cb58b452a38aca45ec3682c5d493a` | `1dbbade056348693390507d591aeef15` |
| `single_rising_bubble/trackedBubInfo_BEM.txt.xz` | 201,298 | 12,352,175 | `12033dc97fa51fb764c1899de4ef61cf` | `e2c09f9afdec3bc3d3ed5a75e68198ed` |
| `single_rising_bubble/trackedBubInfo_Minnaert.txt.xz` | 201,298 | 11,770,134 | `f2a9ebccbb4ce9bc5daa60977ca3ef39` | `80057cea83851d08f4189b9e8ed0e6d6` |
| `single_rising_bubble/trackedBubInfo_NN.txt.xz` | 201,298 | 12,351,993 | `b73c513e0ccd7b8f0452cc77d7b1e214` | `352bc59a14d4c18ca2fced963b86d77a` |
| `fruit_splash/trackedBubInfo_BEM.txt.xz` | 6,359,070 | 408,718,781 | `c76b6c7b150cf2a9be9025ee072a1cce` | `1a6276a01169b0ae8655e4346524d6f9` |
| `fruit_splash/trackedBubInfo_Minnaert.txt.xz` | 6,359,070 | 388,551,793 | `9ae3097deb876b9055c6604889a905a6` | `8fa9589cb8d319a71a3aa8d16efe6fd2` |
| `fruit_splash/trackedBubInfo_NN.txt.xz` | 6,359,070 | 408,642,295 | `259f89fa103a13a9d57eaed1081753be` | `a8c557ab34b62c10966522ee4e8a6602` |
| `exhalation/trackedBubInfo_BEM.txt.xz` | 21,222,779 | 1,321,234,256 | `dfd65e7748aa9f405182e92f355cfd4c` | `23f47e7e0474ed186472f6bbb15c7d29` |
| `exhalation/trackedBubInfo_Minnaert.txt.xz` | 21,222,779 | 1,253,228,293 | `57455e6447ff955a7266716352de095c` | `4fb22684a7f1471f2459be48765aaae7` |
| `exhalation/trackedBubInfo_NN.txt.xz` | 21,222,779 | 1,321,310,331 | `4a79ec6f3e11c8a18fcd964120569763` | `d5ac121c40fb638cb4354f0e87bd15f2` |
