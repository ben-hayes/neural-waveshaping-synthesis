while IFS= read -r args; do
  for inst in vn fl tpt; do
    python scripts/extra_exps.py --instrument $inst \
      $args
  done
done < $1
