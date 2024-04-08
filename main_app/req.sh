# FRONT CCCD
curl -X 'POST' \
  'https://aiclub.uit.edu.vn/namnh/ekyc/cardreader/api/v1/idcardreader' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@assets/front-cccd.jpg;type=image/jpeg' \
  -F 'type_card=0' \
  -F 'is_front=1'

# BACK CCCD
curl -X 'POST' \
  'https://aiclub.uit.edu.vn/namnh/ekyc/cardreader/api/v1/idcardreader' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@assets/back-cccd.jpg;type=image/jpeg' \
  -F 'type_card=0' \
  -F 'is_front=0'

# FRONT CMND
curl -X 'POST' \
  'https://aiclub.uit.edu.vn/namnh/ekyc/cardreader/api/v1/idcardreader' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@assets/front-cmtnd.jpg;type=image/jpeg' \
  -F 'type_card=1' \
  -F 'is_front=1'

# BACK CMND
curl -X 'POST' \
  'https://aiclub.uit.edu.vn/namnh/ekyc/cardreader/api/v1/idcardreader' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@assets/back-cmtnd.jpg;type=image/jpeg' \
  -F 'type_card=1' \
  -F 'is_front=0'