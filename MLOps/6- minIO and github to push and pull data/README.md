#### میخواهیم وقتی کدمون رو پوش میکنیم بره خودش دیتارو از MinIO بگیره و بعد مراحل بعدی رو توی CI/CD انجام بده

- ###### CREATE secret key
  - ###### 1- REPO
  - ###### 2- Setttings
  - ###### 3- Secrets and variables
    - ###### New repository secret --> Define URL, Username and password to connect MINIO
  - ###### 4- Inside the `ci.yml` --> ${{secrets....}}
    - ###### در آخرین مرحله میایم و مدل رو داخل minio ذخیره میکنیم
