def fetch_and_prepare(ticker=TICKER, period=PERIOD):
    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker or internet.")
    df = df[['Open','High','Low','Close','Volume']].dropna()

    # Tambah fitur teknikal
    df['return1'] = df['Close'].pct_change(1)
    df['ma7'] = df['Close'].rolling(7).mean()
    df['ma21'] = df['Close'].rolling(21).mean()
    df['std7'] = df['Close'].rolling(7).std()
    df = df.dropna()

    features = ['Close','Open','High','Low','Volume','return1','ma7','ma21','std7']
    data = df[features].values
    targets = df['Close'].shift(-HORIZON).values

    # Gunakan index eksplisit untuk menghindari mismatch
    valid_idx = np.where(~np.isnan(targets))[0]
    data = data[valid_idx, :]
    targets = targets[valid_idx]

    print("DEBUG shapes → data:", data.shape, "targets:", targets.shape)  # 👀 cek dimensi

    X, y = [], []
    for i in range(SEQ_LEN, len(data)):
        X.append(data[i-SEQ_LEN:i])
        y.append(targets[i])
    return np.array(X), np.array(y), df
