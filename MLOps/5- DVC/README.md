##### 1- Installation --> `python -m pip install dvc`

##### 2- Setup --> `python -m dvc init`

##### 3- Push `.dvc` and `.dvcignore`on git --> `git add .dvc .dvcignore`

##### 4- Add `data` folder to dvc --> `python -m dvc add .\data\... .csv`

- ###### خودش یکسزی اطلاعات به فولدر دیتا اضافه میکنه

##### 5- Push `data` folder on git --> `git add data`

##### 6- Push everything else on git --> `git add .`

###### Merge minIO with DVC to store the data inside the backet

- ###### Create a backet inside the minIo --> name: `datastorage`
- ###### `python -m pip install dvc_s3`
- ###### `python -m dvc remote add minio s3://datastorage`
  - ###### minio is a custom name for remote repository. we add all data using that name.
- ###### `python -m dvc remote modify minio endpointurl http://localhost:9000
  - ###### `endpoint-url` or `endpointurl`
- ###### Setting username and password to access minIO
  - ###### `python -m dvc remote modify --local minio access_key_id simchi`
  - ###### `python -m dvc remote modify --local minio secret_access_key Hossein!7175`
- ###### `git add .`
  - ###### تمامی اطلاعات به جز یوزرنیم و پسورد داخل گیت اضافه میشن چون اونارو نمیخواهیم داخل گیت داشته باشیم و دست همه بیفته
- ###### `python -m dvc push -r minio` --> به ریپوی minio اضافش کن

- ###### If someone else wants to download the data:
  - ###### Before using the below command, we need to Setting username and password using the mentioned commands
  - ###### `python -m dvc pull -r minio`

