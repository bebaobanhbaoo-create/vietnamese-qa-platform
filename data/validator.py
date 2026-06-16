class DataValidator:
    def validate(self, data):
        return len(data) > 0

    def check_columns(self, df, cols):
        return all(c in df.columns for c in cols)

    def check_nulls(self, df):
        return df.isnull().sum().sum() == 0
